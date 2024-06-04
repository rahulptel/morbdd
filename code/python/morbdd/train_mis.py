import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import TrainingHelper
from morbdd.utils.mis import CustomCollater
from morbdd.utils.mis import get_checkpoint_path
from morbdd.utils.mis import get_instance_data
from morbdd.utils.mis import get_size
from torch.distributed import init_process_group
import os
from morbdd import machine


def compute_batch_stats(labels, preds):
    # True positive: label=1 and class=1
    tp = (preds[labels == 1] == 1).sum()
    # True negative: label=0 and class=0
    tn = (preds[labels == 0] == 0).sum()
    # False positive: label=0 and class=1
    fp = (preds[labels == 0] == 1).sum()
    # False negative: label=1 and class=0
    fn = (preds[labels == 1] == 0).sum()

    pos = labels.sum()
    neg = labels.shape[0] - pos
    # items = pos + neg
    return tp, tn, fp, fn, pos, neg


def compute_meta_stats(stats):
    loss, tp, tn, fp, fn, n_pos, n_neg = (stats["loss"], stats["tp"], stats["tn"], stats["fp"], stats["fn"],
                                          stats["n_pos"], stats["n_neg"])

    loss = loss / (n_pos + n_neg)
    acc = ((tp + tn) / (tp + fp + tn + fn))
    f1 = tp / (tp + (0.5 * (fn + fp)))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    stats.update({
        "loss": loss, "acc": acc, "f1": f1, "precision": precision, "recall": recall, "specificity": specificity
    })

    return stats


def print_stats(split, stats):
    print_str = "{}:{}: F1: {:4f}, Acc: {:.4f}, Loss {:.4f}, Recall: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, "
    print_str += "Epoch Time: {:.4f}, Batch Time: {:.4f}, Data Time: {:.4f}"
    ept, bt, dt = -1, -1, -1
    if split == "train":
        ept, bt, dt = stats["epoch_time"], stats["batch_time"], stats["data_time"]

    print(print_str.format(stats["epoch"], split, stats["f1"], stats["acc"], stats["loss"], stats["recall"],
                           stats["precision"], stats["specificity"], ept, bt, dt))


def compute_meta_stats_and_print(master, epoch, split, stats_lst, stats):
    if master:
        meta_stats = compute_meta_stats(stats)
        meta_stats.update({"epoch": epoch + 1})
        stats_lst.append(meta_stats)
        print_stats(split, stats_lst[-1])


class MISBDDNodeDataset(Dataset):
    def __init__(self, bdd_node_dataset, obj, adj, top_k=5, norm_const=100, max_objs=10):
        super(MISBDDNodeDataset, self).__init__()
        self.norm_const = norm_const
        self.max_objs = max_objs

        self.nodes = torch.from_numpy(bdd_node_dataset.astype('int16'))
        perm = torch.randperm(self.nodes.shape[0])
        self.nodes = self.nodes[perm]

        self.top_k = top_k
        self.obj, self.adj = torch.from_numpy(obj), torch.from_numpy(adj)
        self.append_obj_id()
        self.pos = self.precompute_pos_enc(top_k, self.adj)

    def __getitem__(self, item):
        pid = self.nodes[item, 0]

        return (self.obj[pid],
                self.adj[pid],
                self.pos[pid] if self.top_k > 0 else None,
                self.nodes[item, 0],
                self.nodes[item, 1],
                self.nodes[item, 2],
                self.nodes[item, 3:103],
                self.nodes[item, 103])

    def __len__(self):
        return self.nodes.shape[0]

    @staticmethod
    def precompute_pos_enc(top_k, adj):
        p = None
        if top_k > 0:
            # Calculate position encoding
            U, S, Vh = torch.linalg.svd(adj)
            U = U[:, :, :top_k]
            S = (torch.diag_embed(S)[:, :top_k, :top_k]) ** (1 / 2)
            Vh = Vh[:, :top_k, :]

            L, R = U @ S, S @ Vh
            R = R.permute(0, 2, 1)
            p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)

        return p

    def append_obj_id(self):
        n_items, n_objs, n_vars = self.obj.shape
        obj_id = torch.arange(1, n_objs + 1) / self.max_objs
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((n_items, 1, 1))
        # n_items x n_objs x n_vars x 2
        self.obj = torch.cat((self.obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)


class MISTrainingHelper(TrainingHelper):
    def __init__(self, cfg):
        super(MISTrainingHelper, self).__init__()
        self.cfg = cfg

    def get_dataset(self, split, from_pid, to_pid):
        bdd_node_dataset = np.load(str(path.dataset) + f"/{self.cfg.prob.name}/{self.cfg.size}/{split}.npy")
        valid_rows = (from_pid <= bdd_node_dataset[:, 0])
        valid_rows &= (bdd_node_dataset[:, 0] < to_pid)

        bdd_node_dataset = bdd_node_dataset[valid_rows]
        if split == "val":
            bdd_node_dataset[:, 0] -= 1000

        obj, adj = [], []
        for pid in range(from_pid, to_pid):
            data = get_instance_data(self.cfg.size, split, pid)
            obj.append(data["obj_coeffs"])
            adj.append(data["adj_list"])
        obj, adj = np.array(obj), np.stack(adj)
        dataset = MISBDDNodeDataset(bdd_node_dataset, obj, adj, top_k=self.cfg.top_k)

        return dataset

    def get_dataloader(self, split, from_pid, to_pid, shuffle=True, drop_last=False):
        bdd_node_dataset = np.load(str(path.dataset) + f"/{self.cfg.prob.name}/{self.cfg.size}/{split}.npy")
        valid_rows = (from_pid <= bdd_node_dataset[:, 0])
        valid_rows &= (bdd_node_dataset[:, 0] <= to_pid)

        bdd_node_dataset = bdd_node_dataset[valid_rows]
        if split == "val":
            bdd_node_dataset[:, 0] -= 1000

        obj, adj = [], []
        for pid in range(from_pid, to_pid):
            data = get_instance_data(self.cfg.size, split, pid)
            obj.append(data["obj_coeffs"])
            adj.append(data["adj_list"])
        obj, adj = np.array(obj), np.stack(adj)

        dataset = MISBDDNodeDataset(bdd_node_dataset, obj, adj, top_k=self.cfg.top_k)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=self.cfg.n_worker_dataloader)

        return len(dataset), dataloader


def train(dataloader, model, optimizer, device, clip_grad=1.0, norm_type=2.0):
    data_time, batch_time, epoch_time = 0.0, 0.0, 0.0
    losses = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    n_pos, n_neg = 0, 0

    model.train()
    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100
        data_time += time.time() - start_time

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Learn
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=norm_type)
        optimizer.step()

        # Compute stats
        preds = torch.argmax(F.softmax(logits.detach(), dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = compute_batch_stats(labels.cpu().numpy(),
                                                                 preds.detach().cpu().numpy())

        losses += (loss.item() * objs.shape[0])
        tp += _tp
        tn += _tn
        fp += _fp
        fn += _fn
        n_pos += _n_pos
        n_neg += _n_neg
        batch_time += time.time() - start_time
        start_time = time.time()

    data_time /= len(dataloader)
    batch_time /= len(dataloader)
    losses /= (n_pos + n_neg)

    return {"loss": losses, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_pos": n_pos, "n_neg": n_neg,
            "data_time": data_time, "batch_time": batch_time, "epoch_time": epoch_time}


@torch.no_grad()
def validate(dataloader, model, device):
    losses = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    n_pos, n_neg = 0, 0

    model.eval()
    for batch in dataloader:
        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = compute_batch_stats(labels.cpu().numpy(),
                                                                 preds.detach().cpu().numpy())
        losses += (loss.item() * objs.shape[0])
        tp += _tp
        tn += _tn
        fp += _fp
        fn += _fn
        n_pos += _n_pos
        n_neg += _n_neg

    losses /= (n_pos + n_neg)

    return {"loss": losses, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_pos": n_pos, "n_neg": n_neg,
            "data_time": None, "batch_time": None, "epoch_time": None}


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)
    ckpt_path = get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)
    print("Checkpoint path: {}".format(ckpt_path))

    # Set-up device
    device_str = "cpu"
    pin_memory = False
    if torch.cuda.is_available():
        device_str = "cuda"
        pin_memory = True
    device = torch.device(device_str)
    print("Device :", device)
    train_stats_lst, val_stats_lst = [], []

    # Initialize train data loader
    print("N worker dataloader: ", cfg.n_worker_dataloader)
    train_dataset = helper.get_dataset("train",
                                       cfg.dataset.train.from_pid,
                                       cfg.dataset.train.to_pid)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.n_worker_dataloader,
                                  pin_memory=pin_memory)
    val_dataset = helper.get_dataset("val",
                                     cfg.dataset.val.from_pid,
                                     cfg.dataset.val.to_pid)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.n_worker_dataloader,
                                pin_memory=pin_memory)
    print("Train samples: {}, Val samples {}".format(len(train_dataset), len(val_dataset)))
    print("Train loader: {}, Val loader {}".format(len(train_dataloader), len(val_dataloader)))

    best_f1 = 0
    model = ParetoStatePredictorMIS(encoder_type="transformer",
                                    n_node_feat=2,
                                    n_edge_type=2,
                                    d_emb=cfg.d_emb,
                                    top_k=cfg.top_k,
                                    n_blocks=cfg.n_blocks,
                                    n_heads=cfg.n_heads,
                                    dropout_token=cfg.dropout_token,
                                    dropout=cfg.dropout,
                                    bias_mha=cfg.bias_mha,
                                    bias_mlp=cfg.bias_mlp,
                                    h2i_ratio=cfg.h2i_ratio,
                                    device=device).to(device)
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)

    # Reset training stats per epoch
    for epoch in range(cfg.epochs):
        start_time = time.time()
        stats = train(train_dataloader, model, optimizer, device, clip_grad=cfg.clip_grad, norm_type=cfg.norm_type)
        stats["epoch_time"] = time.time() - start_time
        compute_meta_stats_and_print(True, epoch, "train", train_stats_lst, stats)

        if (epoch + 1) % cfg.validate_every == 0:
            stats = validate(val_dataloader, model, device)
            compute_meta_stats_and_print(True, epoch, "val", val_stats_lst, stats)

        best_model = False
        if val_stats_lst[-1]["f1"] > best_f1:
            best_f1 = val_stats_lst[-1]["f1"]
            best_model = True
            print("***{} Best f1: {}".format(epoch + 1, best_f1))

        # helper.save(epoch,
        #             ckpt_path,
        #             best_model=best_model,
        #             model=model.state_dict(),
        #             optimizer=optimizer.state_dict())

        print()


if __name__ == "__main__":
    main()
