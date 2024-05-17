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

    def get_dataloader(self, split, from_pid, to_pid, shuffle=True, drop_last=False):
        bdd_node_dataset = np.load(str(path.dataset) + f"/{self.cfg.prob.name}/{self.cfg.size}/{split}.npy")
        valid_rows = (from_pid <= bdd_node_dataset[:, 0])
        valid_rows &= (bdd_node_dataset[:, 0] <= to_pid)

        bdd_node_dataset = bdd_node_dataset[valid_rows]

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


def train_batch(epoch, batch, model, optimizer, device, helper):
    objs, adjs, pos, pids, lids, vids, states, labels = batch
    objs = objs / 100

    # Get logits
    logits = model(objs.to(device),
                   adjs.to(device),
                   pos.to(device),
                   lids.to(device),
                   vids.to(device),
                   states.to(device))

    # Compute loss
    logits, labels = logits.reshape(-1, 2), labels.to(device).long().reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

    # Learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute stats
    preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
    batch_stats = helper.compute_batch_stats(labels, preds)
    batch_stats["loss"] = loss.item()

    return batch_stats


@torch.no_grad()
def validate(epoch, dataloader, model, device, helper):
    stats = helper.reset_epoch_stats()

    model.eval()
    for batch in dataloader:
        start = time.time()
        objs, adjs, pos, pids, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits
        logits = model(objs.to(device),
                       adjs.to(device),
                       pos.to(device),
                       lids.to(device),
                       vids.to(device),
                       states.to(device))
        # Compute loss
        logits, labels = logits.reshape(-1, 2), labels.to(device).long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        end_time = time.time() - start

        labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
        batch_stats = helper.compute_batch_stats(labels, preds)
        batch_stats["loss"] = loss.item()
        batch_stats["time"] = end_time
        helper.update_running_stats(stats, batch_stats)

    helper.compute_epoch_stats(stats)
    helper.print_stats(epoch, stats, split="Val")

    model.train()

    return stats


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = MISTrainingHelper(cfg)
    ckpt_path = get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    n_train, train_dataloader = helper.get_dataloader("train", 0, 1000, drop_last=True)
    n_val, val_dataloader = helper.get_dataloader("val", 1000, 1100, drop_last=False)
    print("Train samples: {}, Val samples {}".format(n_train, n_val))

    best_f1 = 0
    model = ParetoStatePredictorMIS(encoder_type="transformer",
                                    n_node_feat=2,
                                    n_edge_type=2,
                                    d_emb=cfg.d_emb,
                                    top_k=cfg.top_k,
                                    n_blocks=cfg.n_blocks,
                                    n_heads=cfg.n_heads,
                                    dropout_token=cfg.dropout_token,
                                    bias_mha=cfg.bias_mha,
                                    dropout=cfg.dropout,
                                    bias_mlp=cfg.bias_mlp,
                                    h2i_ratio=cfg.h2i_ratio,
                                    device=device).to(device)
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr)

    # Reset training stats per epoch
    stats = helper.reset_epoch_stats()
    for epoch in range(cfg.epochs):
        for batch_id, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch_stats = train_batch(epoch, batch, model, optimizer, device, helper)
            batch_stats["time"] = time.time() - start_time
            helper.update_running_stats(stats, batch_stats)
            if cfg.logging > 1:
                helper.print_batch_stats(epoch, batch_id, batch_stats)

        helper.compute_epoch_stats(stats)
        helper.print_stats(epoch, stats)
        helper.add_epoch_stats("train", (epoch + 1, stats))

        best_model = False
        if (epoch + 1) % cfg.validate_every == 0:
            val_stats = validate(epoch, val_dataloader, model, device, helper)
            helper.add_epoch_stats("val", (epoch, val_stats))
            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                best_model = True
                print("***{} Best f1: {}".format(epoch + 1, val_stats["f1"]))

        helper.save(epoch,
                    ckpt_path,
                    best_model=best_model,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict())


if __name__ == "__main__":
    main()
