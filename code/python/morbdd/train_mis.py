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
        self.X, self.Y, self.orders, self.masks, self.obj_coeffs, self.weights = [], [], [], [], [], []
        self.id_to_inst = []

        for pid in range(from_pid, to_pid):
            file = prefix + f"/{pid}.pt"
            idata = read_from_zip(archive, file, format="pt")
            x, y, order, obj_coeffs = idata["x"], idata["y"], idata["order"], idata["obj_coeffs"]
            weight = x[:, 0]

            lids = x[:, 1].numpy().astype(int)
            mask = np.ones((x.shape[0], cfg.prob.n_vars))
            for l, m in zip(lids, mask):
                m[order[:l + 1]] = 0
            mask = torch.from_numpy(mask)

            self.weights.append(weight)
            self.X.append(x[:, 1:])
            self.Y.append(y)
            self.orders.append(order)
            self.masks.append(mask)
            self.obj_coeffs.append(obj_coeffs)
            self.id_to_inst.extend([pid] * x.shape[0])

        self.X = torch.cat(self.X, dim=0).float()
        self.Y = torch.cat(self.Y, dim=0).float()
        self.masks = torch.cat(self.masks, dim=0).float()
        self.orders = torch.stack(self.orders, dim=0).float()
        self.obj_coeffs = torch.stack(self.obj_coeffs, dim=0).float()
        self.obj_coeffs = self.obj_coeffs.permute(0, 2, 1)
        self.weights = torch.cat(self.weights, dim=0).float().reshape(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        pid = self.id_to_inst[i]
        return self.X[i], self.Y[i], self.masks[i], self.obj_coeffs[pid]

    def __str__(self):
        text = f"X:  {self.X.shape}\n"
        text += f"Y: {self.Y.shape}\n"
        text += f"Mask: {self.masks.shape}\n"
        text += f"orders: {self.orders.shape}\n"
        text += f"obj_coeffs: {self.obj_coeffs.shape}\n"

        return text


class NodeEncoder(nn.Module):
    def __init__(self, n_features):
        super(NodeEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, 3 * 128)
        self.encX = nn.Linear(3 * 128, 128)
        self.encPN = nn.Linear(3 * 128, 128)
        self.encNN = nn.Linear(3 * 128, 128)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(128 * 3)

    def forward(self, x, pos_adj):
        neg_adj = 1 - pos_adj

        x = self.relu(self.linear1(x))

        x1 = self.encX(x)
        x2 = pos_adj @ self.encPN(x)
        x3 = neg_adj @ self.encNN(x)
        x = torch.cat((x1, x2, x3), dim=2)

        x = self.layer_norm(x)

        return x


class ParetoNodePredictor(nn.Module):
    def __init__(self, cfg):
        super(ParetoNodePredictor, self).__init__()
        self.enc1 = nn.Linear(3 * 128, 64)
        self.enc2 = nn.Linear(128, 64)
        self.enc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layer_encoding = nn.Parameter(torch.randn(cfg.prob.n_vars, cfg.emb_dim))

    def forward(self, node, mask, coeff, adj):
        mask = mask.unsqueeze(2)
        var_feat = torch.cat((coeff, mask), dim=-1)

        vars_enc = self.variable_encoder(var_feat, adj)

        layer_enc = self.layer_encoding[node[:, 1].view(-1).int()]
        state_enc = todo
        var_enc = todo

        # ve = self.relu(self.enc1(ve))
        # se = self.relu(self.enc1(se))
        # x = self.sigmoid(self.enc2(torch.cat((ve, se), dim=-1)))

        return node


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
        x, y, mask, coeffs = batch
        print(x.shape, y.shape, mask.shape, coeffs.shape)
        predictor(x, mask, coeffs, x)
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
