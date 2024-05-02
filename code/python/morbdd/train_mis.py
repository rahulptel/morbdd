import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip
from morbdd.generate_mis_dataset import get_node_data

random.seed(42)
rng = np.random.RandomState(42)
seeds = rng.randint(1, 10000, 1000)


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


obj_coeffs = {
    "train": {},
    "val": {}
}

adj = {
    "train": {},
    "val": {}
}

node_data = {
    "train": {},
    "val": {}
}


def get_node_data(seed, n_vars, bdd, with_parent=False):
    random.seed(seed)
    data_lst = []
    labels_lst = []
    counts = []
    n_total = 0
    for lid, layer in enumerate(bdd):
        neg_data_lst = []
        pos_data_lst = []
        for nid, node in enumerate(layer):
            # Binary state of the current node
            state = np.zeros(n_vars)
            state[node['s']] = 1
            node_data = np.concatenate(([lid], state))

            if node['score'] > 0:
                pos_data_lst.append(node_data)
            else:
                neg_data_lst.append(node_data)

        # Get label
        n_pos = len(pos_data_lst)
        labels = [1] * n_pos
        # data_lst.extend(pos_data_lst)

        # Undersample negative class
        n_neg = min(len(neg_data_lst), n_pos)
        labels.extend([0] * n_neg)
        random.shuffle(neg_data_lst)
        neg_data_lst = neg_data_lst[:n_neg]
        # data_lst.extend(neg_data_lst)
        pos_data_lst.extend(neg_data_lst)

        data_lst.append(pos_data_lst)
        labels_lst.append(labels)


def get_mis_dataset_instance(size, split, pid):
    global obj_coeffs, adj
    archive_bdds = path.bdd / f"indepset/{size}.zip"

    # Read instance data
    data = get_instance_data("indepset", size, split, pid)
    file = f"{size}/{split}/{pid}.json"
    bdd = read_from_zip(archive_bdds, file, format="json")
    # Read order
    order = path.order.joinpath(f"indepset/{size}/{split}/{pid}.dat").read_text()
    order = np.array(list(map(int, order.strip().split())))
    # Get node data
    if pid not in obj_coeffs[split]:
        obj_coeffs[split][pid] = torch.from_numpy(np.array(data["obj_coeffs"]))
    if pid not in adj[split]:
        adj[split][pid] = torch.from_numpy(np.array(data["adj_list"]))
    X, Y = get_node_data(cfg, bdd)
    X, Y, order = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(order)


def get_mis_dataset(size, split):
    pass


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
    val_dataset = MISDataset(cfg, split="val")
    print("Train samples {}, Val samples {}".format(len(train_dataset), len(val_dataset)))

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Build model
    # model = ParetoNodePredictor(cfg)

    # Initialize optimizer
    # opt_cls = getattr(optim, cfg.opt.name)
    # optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr)

    for epoch in range(cfg.epochs):
        train_dataset = MISDataset(cfg, split="train")
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  sampler=WeightedRandomSampler(train_dataset.weights,
                                                                num_samples=len(train_dataset),
                                                                replacement=True))

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
