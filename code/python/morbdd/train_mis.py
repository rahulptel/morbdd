import random

import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip
import torch.optim as optim

random.seed(42)


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


class MISDataset(Dataset):
    def __init__(self, cfg, split="train"):
        from_pid, to_pid = cfg[split].from_pid, cfg[split].to_pid
        archive = path.dataset / f"{cfg.prob.name}/{cfg.size}.zip"
        dtype = f"{cfg.layer_weight}-{cfg.neg_to_pos_ratio}-"
        dtype += "parent" if cfg.with_parent else "no-parent"

        prefix = f"{cfg.size}/{split}/{dtype}"
        self.X, self.Y, self.weights = [], [], []
        self.adj, self.obj_coeffs, self.orders, self.masks, self.var_idxs = [], [], [], [], []
        self.id_to_inst = []

        for pid in range(from_pid, to_pid):
            file = prefix + f"/{pid}.pt"
            idata = read_from_zip(archive, file, format="pt")
            x, y, adj, order, obj_coeffs = idata["x"], idata["y"], idata["adj"], idata["order"], idata["obj_coeffs"]

            # Node weight
            weight = x[:, 0]

            # Layer in which the node belongs
            lids = x[:, 1].numpy().astype(int)
            mask = np.ones((x.shape[0], cfg.prob.n_vars))
            for l, m in zip(lids, mask):
                m[order[:l + 1]] = 0
                self.var_idxs.append(order[l + 1])
            mask = torch.from_numpy(mask)

            # n_nodes x n_node_features_0
            self.X.append(x[:, 1:])
            # n_nodes
            self.Y.append(y)
            # n_nodes
            self.weights.append(weight)
            # n_nodes x n_vars_mis
            self.masks.append(mask)
            # n_nodes
            self.id_to_inst.extend([pid] * x.shape[0])

            # n_vars_mip x n_vars_mip
            self.adj.append(adj)
            # n_vars_mip
            self.orders.append(order)
            # n_objs x n_vars_mip
            self.obj_coeffs.append(obj_coeffs)

        self.X = torch.cat(self.X, dim=0).float()
        self.Y = torch.cat(self.Y, dim=0).float()
        self.weights = torch.cat(self.weights, dim=0).float().reshape(-1)
        self.masks = torch.cat(self.masks, dim=0).float()
        self.var_idxs = torch.from_numpy(np.array(self.var_idxs)).float()

        self.adj = torch.stack(self.adj, dim=0).float()
        self.orders = torch.stack(self.orders, dim=0).float()
        self.obj_coeffs = torch.stack(self.obj_coeffs, dim=0).float()
        self.obj_coeffs = self.obj_coeffs.permute(0, 2, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        pid = self.id_to_inst[i]
        return self.X[i], self.Y[i], self.masks[i], self.adj[pid], self.obj_coeffs[pid], self.var_idxs[i]

    def __str__(self):
        text = f"X:  {self.X.shape}\n"
        text += f"Y: {self.Y.shape}\n"
        text += f"Mask: {self.masks.shape}\n"
        text += f"orders: {self.orders.shape}\n"
        text += f"obj_coeffs: {self.obj_coeffs.shape}\n"

        return text


class FeedForwardUnit(nn.Module):
    def __init__(self, ni, nh, no, dropout=0.0, bias=True, activation="relu"):
        super(FeedForwardUnit, self).__init__()
        self.i2h = nn.Linear(ni, nh, bias=bias)
        self.h2o = nn.Linear(nh, no, bias=bias)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.i2h(x)
        x = self.act(x)
        x = self.h2o(x)
        x = self.dropout(x)

        return x


class GraphForwardUnit(nn.Module):
    def __init__(self, ni, nh, no, dropout=0.0, bias=True, activation="relu"):
        super(GraphForwardUnit, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(ni)
        self.ffu = FeedForwardUnit(ni, nh, no, dropout=dropout, bias=bias, activation=activation)

        self.i2o = None
        if ni != no:
            self.i2o = nn.Linear(ni, no, bias=False)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()

        self.ln_2 = nn.LayerNorm(no)
        self.pos = nn.Linear(no, no, bias)

        self.ln_3 = nn.LayerNorm(no)
        self.neg = nn.Linear(no, no, bias)
        self.combine = nn.Linear(2 * no, no, bias)

    def forward(self, x, adj):
        neg_adj = 1 - adj

        ffu_x = self.ffu(self.ln_1(x))
        x = x if self.i2o is None else self.i2o(x)
        x = x + ffu_x

        x_pos = self.act(self.pos(self.ln_2(adj @ x)))
        x_neg = self.act(self.neg(self.ln_3(neg_adj @ x)))
        x = x + self.act(self.combine(self.dropout(torch.cat((x_pos, x_neg), dim=-1))))

        return x


class GraphVariableEncoder(nn.Module):
    def __init__(self, cfg):
        super(GraphVariableEncoder, self).__init__()
        self.cfg = cfg
        self.n_layers = self.cfg.graph_enc.n_layers
        n_feat = self.cfg.graph_enc.n_feat
        n_emb = self.cfg.graph_enc.n_emb
        act = self.cfg.graph_enc.activation
        dp = self.cfg.graph_enc.dropout
        bias = self.cfg.graph_enc.bias

        self.units = nn.ModuleList()
        for layer in range(self.n_layers):
            if layer == 0:
                self.units.append(GraphForwardUnit(n_feat, n_emb, n_emb, dropout=dp, activation=act, bias=bias))
            else:
                self.units.append(GraphForwardUnit(n_emb, n_emb, n_emb, dropout=dp, activation=act, bias=bias))

    def forward(self, x, adj):
        for layer in range(self.n_layers):
            x = self.units[layer](x, adj)

        return x


#
# class StateEncoder(nn.Module):
#     def __init__(self, n_features):
#         super(StateEncoder, self).__init__()
#         self.linear1 = nn.Linear(n_features, 3 * 128)
#         self.encX = nn.Linear(3 * 128, 128)
#         self.encPN = nn.Linear(3 * 128, 128)
#         self.encNN = nn.Linear(3 * 128, 128)
#         self.relu = nn.ReLU()
#         self.layer_norm = nn.LayerNorm(128 * 3)
#
#     def forward(self, x, pos_adj):
#         neg_adj = 1 - pos_adj
#
#         x = self.relu(self.linear1(x))
#
#         x1 = self.encX(x)
#         x2 = pos_adj @ self.encPN(x)
#         x3 = neg_adj @ self.encNN(x)
#         x = torch.cat((x1, x2, x3), dim=2)
#
#         x = self.layer_norm(x)
#
#         return x


class ParetoNodePredictor(nn.Module):
    def __init__(self, cfg):
        super(ParetoNodePredictor, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.variable_encoder = GraphVariableEncoder(cfg)
        # self.state_encoder = StateEncoder(cfg)
        self.layer_encoding = nn.Parameter(torch.randn(cfg.prob.n_vars, cfg.n_emb))

        self.ln = nn.LayerNorm(cfg.n_emb)
        if cfg.agg == "sum":
            self.linear1 = nn.Linear(cfg.n_emb, 1)
        if cfg.agg == "cat":
            self.linear1 = nn.Linear(4 * cfg.n_emb, cfg.n_emb)

    def forward(self, node_feat, var_feat, adj, var_id):
        print(node_feat.shape, var_feat.shape, adj.shape, var_id.shape)

        var_enc = self.variable_encoder(var_feat, adj)  # B x n_var_mis x n_emb
        B, nV, nF = var_enc.shape

        graph_enc = var_enc.sum(dim=1)  # B x n_emb
        layer_var_enc = var_enc[torch.arange(B).unsqueeze(1), var_id.unsqueeze(1).int(), :].squeeze(1)  # B x n_emb
        layer_enc = self.layer_encoding[node_feat[:, 0].view(-1).int()]

        print(type(node_feat[0][1:].int()))
        print(var_enc.shape, var_enc[0].shape, node_feat.shape)

        print(node_feat[0][1:].int())
        print(var_enc[0, node_feat[0, 1:].int(), :].shape)

        state_enc = torch.stack([ve[node_feat[i, 1:].int()].sum(dim=1)
                                 for i, ve in enumerate(var_enc)])
        print(state_enc.shape)
        #
        # x = None
        # if self.cfg.agg == "sum":
        #     x = graph_enc + layer_var_enc + layer_enc + state_enc
        # elif self.cfg.agg == "cat":
        #     x = torch.cat((graph_enc, layer_var_enc, layer_enc, state_enc), dim=-1)
        #
        # self.relu(self.linear1(self.dropout(self.ln(x))))
        # return x


def pad_samples(samples, max_parents, n_vars):
    for sample in samples:
        if len(sample[2]) < max_parents:
            sample[2] = np.vstack((sample[2],
                                   np.zeros((max_parents - len(sample[2]), n_vars))
                                   ))

        if len(sample[3]) < max_parents:
            sample[3] = np.vstack((sample[3],
                                   np.zeros((max_parents - len(sample[3]), n_vars))
                                   ))

    return samples


def order2rank(order):
    rank = np.zeros_like(order)
    for r, item in enumerate(order):
        rank[item] = r + 1

    rank = rank / len(order)
    return rank


def reorder_data(data, order):
    return data


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    # Get dataset and dataloader
    train_dataset = MISDataset(cfg, split="train")
    val_dataset = MISDataset(cfg, split="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              sampler=WeightedRandomSampler(train_dataset.weights,
                                                            num_samples=len(train_dataset),
                                                            replacement=True))
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    predictor = ParetoNodePredictor(cfg)
    for i, batch in enumerate(train_loader):
        nf, y, m, adj, ocoeff, vidx = batch

        mask = m.unsqueeze(2)
        vf = torch.cat((ocoeff, mask), dim=-1)

        predictor(nf, vf, adj, vidx)

        break
    # Build model

    # Initialize optimizer
    # opt_cls = getattr(optim, cfg.opt.name)
    # opt_cls(model.params(), lr=cfg.opt.lr)

    # encoder = NodeEncoder(4)
    # node_predictor = ParetoNodePredictor()
    # for epoch in range(cfg.train.epochs):
    #     random.shuffle(train_ids)
    #     for i, train_id in enumerate(train_ids):
    #         archive = path.inst / cfg.prob.name / f"{cfg.size}.zip"
    #         file = f"{cfg.size}/train/ind_7_{cfg.size}_{train_id}.dat"
    #         data = read_instance_indepset(archive, file)
    #
    #         order_path = path.order / cfg.prob.name / cfg.size / "train" / f"{train_id}.dat"
    #         print()
    #
    #         order = list(map(int, order_path.read_text(encoding="utf-8").strip().split(" ")))
    #
    #         bdd_path = path.bdd / cfg.prob.name / cfg.size / "train" / f"{train_id}.json"
    #         bdd = json.load(open(bdd_path, "r"))
    #
    #         pos, neg = get_dataset(bdd)
    #
    #         ranks = []
    #         for p in pos:
    #             rank = np.ones_like(order) * -1.0
    #             for r in range(p[0] + 1):
    #                 rank[order[r]] = (r + 1) / cfg.prob.n_vars
    #             ranks.append(rank)
    #         ranks = np.array(ranks)
    #         ranks = np.expand_dims(ranks, 2)
    #
    #         obj_coeffs = np.array(data["obj_coeffs"])
    #         obj_coeffs = obj_coeffs.T
    #         obj_coeffs = obj_coeffs[np.newaxis, :, :]
    #         obj_coeffs = np.repeat(obj_coeffs, 404, 0)
    #
    #         features = np.concatenate((obj_coeffs, ranks), axis=2)
    #
    #         features_t = torch.from_numpy(features).float()
    #         pos_adj = np.expand_dims(data['adj_list'], 0)
    #         pos_adj = np.repeat(pos_adj, 404, 0)
    #         pos_adj = torch.from_numpy(pos_adj).float()
    #
    #         node_emb = encoder(features_t, pos_adj)
    #         print(node_emb.shape)
    #
    #         variable_emb = []
    #         state_emb = []
    #
    #         for p, ne in zip(pos, node_emb):
    #             variable_emb.append(ne[order[p[0]]])
    #             state_emb.append(T.sum(ne[p[1]], dim=0))
    #
    #         variable_emb = torch.stack(variable_emb, dim=0)
    #         state_emb = torch.stack(state_emb, dim=0)
    #         preds = node_predictor(variable_emb, state_emb)
    #
    #         if i % cfg.val.every == 0:
    #             pass


if __name__ == "__main__":
    main()
