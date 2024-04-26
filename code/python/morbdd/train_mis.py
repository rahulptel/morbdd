import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip

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
        from_pid, to_pid = cfg.dataset.pid[split].start, cfg.dataset.pid[split].end
        with_parent = cfg.dataset.with_parent
        layer_weight = cfg.dataset.layer_weight
        npratio = cfg.dataset.neg_to_pos_ratio

        archive = path.dataset / f"{cfg.prob.name}/{cfg.size}.zip"
        dtype = f"{layer_weight}-{npratio}-"
        dtype += "parent" if with_parent else "no-parent"

        prefix = f"{cfg.size}/{split}/{dtype}"
        self.X, self.Y, self.weights = [], [], []
        self.adj, self.obj_coeffs, self.orders, self.masks, self.var_idxs = [], [], [], [], []
        self.id_to_inst = []

        for i, pid in enumerate(range(from_pid, to_pid)):
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
            self.id_to_inst.extend([i] * x.shape[0])

            # n_vars_mip x n_vars_mip
            self.adj.append(adj)
            # n_vars_mip
            self.orders.append(order)
            # n_objs x n_vars_mip
            self.obj_coeffs.append(obj_coeffs)

        self.X = torch.cat(self.X, dim=0).float()
        self.Y = torch.cat(self.Y, dim=0)
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


class ParetoNodePredictor(nn.Module):
    def __init__(self, cfg):
        super(ParetoNodePredictor, self).__init__()
        self.cfg = cfg
        self.variable_encoder = GraphVariableEncoder(cfg)
        # self.state_encoder = StateEncoder(cfg)
        self.layer_encoding = nn.Parameter(torch.randn(cfg.prob.n_vars, cfg.n_emb))

        self.ff = nn.ModuleList()
        self.ff.append(nn.LayerNorm(cfg.n_emb))
        self.ff.append(nn.Dropout(cfg.dropout))
        if cfg.agg == "sum":
            self.ff.append(nn.Linear(cfg.n_emb, 2))
        if cfg.agg == "cat":
            self.ff.append(nn.Linear(4 * cfg.n_emb, 2))
        self.ff = nn.Sequential(*self.ff)

    def forward(self, node_feat, var_feat, adj, var_id):
        # B x n_var_mis x n_emb
        var_enc = self.variable_encoder(var_feat, adj)

        # B x n_emb
        graph_enc = var_enc.sum(dim=1)

        # B x n_emb
        B, nV, nF = var_enc.shape
        layer_var_enc = var_enc[torch.arange(B), var_id.int(), :].squeeze(1)

        # B x n_emb
        layer_enc = self.layer_encoding[node_feat[:, 0].view(-1).int()]

        # B x n_emb
        active_vars = node_feat[:, 1:].bool()
        state_enc = torch.stack([ve[active_vars[i]].sum(dim=0)
                                 for i, ve in enumerate(var_enc)])

        x = None
        if self.cfg.agg == "sum":
            x = graph_enc + layer_var_enc + layer_enc + state_enc
        elif self.cfg.agg == "cat":
            x = torch.cat((graph_enc, layer_var_enc, layer_enc, state_enc), dim=-1)

        x = self.ff(x)
        return x


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


def train(dataloader, model, optimizer, epoch):
    model.train()
    for i, batch in enumerate(dataloader):
        # Get logits
        nf, y, m, adj, ocoeff, vidx = batch
        mask = m.unsqueeze(2)
        vf = torch.cat((ocoeff, mask), dim=-1)
        logits = model(nf, vf, adj, vidx)

        # Compute loss
        loss = F.cross_entropy(logits, y.view(-1))

        # Learn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch {}, Batch {}, Loss {}".format(epoch, i, loss.item()))


def validate(dataloader, model, epoch, n_samples):
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            nf, y, m, adj, ocoeff, vidx = batch

            mask = m.unsqueeze(2)
            vf = torch.cat((ocoeff, mask), dim=-1)
            logits = model(nf, vf, adj, vidx)
            preds = torch.argmax(logits, dim=-1)
            # print(preds[:5], y[:5], preds[:5] == y[:5])
            # print((preds[:5] == y[:5]), (preds[:5] == y[:5]).numpy().sum())
            accuracy += (preds == y).numpy().sum()

    print("Epoch {} Accuracy: {}".format(epoch, accuracy / n_samples))


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    # Get dataset and dataloader
    train_dataset = MISDataset(cfg, split="train")
    val_dataset = MISDataset(cfg, split="val")
    print("Train samples {}, Val samples {}".format(len(train_dataset), len(val_dataset)))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              sampler=WeightedRandomSampler(train_dataset.weights,
                                                            num_samples=len(train_dataset),
                                                            replacement=True))
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Build model
    model = ParetoNodePredictor(cfg)

    # Initialize optimizer
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr)

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(train_loader):
            # Get logits
            nf, y, m, adj, ocoeff, vidx = batch
            mask = m.unsqueeze(2)
            vf = torch.cat((ocoeff, mask), dim=-1)
            logits = model(nf, vf, adj, vidx)
            # print(logits.shape)
            # print(logits[:5])

            # Compute loss
            loss = F.cross_entropy(logits, y.view(-1))

            # Learn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch {}, Batch {}, Loss {}".format(epoch, i, loss.item()))
            # if i == 10:
            #     break

            if epoch == 0 and i == 0:
                validate(val_loader, model, epoch, n_samples=len(val_dataset))

            # if i % cfg.val_every == 0:
            #     validate(val_loader, model)

        if epoch % cfg.val_every == 0:
            validate(val_loader, model, epoch, n_samples=len(val_dataset))
        # break


if __name__ == "__main__":
    main()
