import ast
import json
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import get_instance_data
from morbdd.utils import get_size
import time

obj, adj = {}, {}


class TrainingHelper:
    def __init__(self):
        self.train_stats = []
        self.val_stats = []

    def add_epoch_stats(self, split, stats):
        stat_lst = getattr(self, f"{split}_stats")
        stat_lst.append(stats)

    @staticmethod
    def get_dataloader(root_path, n_vars, split, start, end, batch_size, num_workers=0, shardshuffle=True,
                       random_shuffle=True):
        dataset_path = root_path + f"/{split}/" + "bdd-layer-{" + f"{start}..{end}" + "}.tar"
        dataset = wds.WebDataset(dataset_path, shardshuffle=shardshuffle).shuffle(5000)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                collate_fn=CustomCollater(n_vars=n_vars, random_shuffle=random_shuffle))

        return dataloader

    @staticmethod
    def reset_epoch_stats():
        return {"items": 0, "loss": 0, "acc": 0, "f1": 0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0, "pos": 0, "neg": 0}

    @staticmethod
    def update_running_stats(curr_stats, new_stats):
        curr_stats["pos"] += new_stats["pos"]
        curr_stats["neg"] += new_stats["neg"]
        curr_stats["items"] += (new_stats["pos"] + new_stats["neg"])
        curr_stats["loss"] += (new_stats["loss"] * (new_stats["pos"] + new_stats["neg"]))
        curr_stats["tp"] += new_stats["tp"]
        curr_stats["fp"] += new_stats["fp"]
        curr_stats["tn"] += new_stats["tn"]
        curr_stats["fn"] += new_stats["fn"]

    @staticmethod
    def compute_epoch_stats(stats):
        stats["loss"] = np.round(stats["loss"] / stats["items"], 3)
        stats["acc"] = np.round(((stats["tp"] + stats["tn"]) /
                                 (stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"])), 3)
        stats["f1"] = np.round(stats["tp"] / (stats["tp"] + (0.5 * (stats["fn"] + stats["fp"]))), 3)
        stats["tp"] = np.round(stats["tp"] / stats["pos"], 3)
        stats["fp"] = np.round(stats["fp"] / stats["neg"], 3)
        stats["tn"] = np.round(stats["tn"] / stats["neg"], 3)
        stats["fn"] = np.round(stats["fn"] / stats["pos"], 3)

    @staticmethod
    def print_batch_stats(epoch, batch_id, stats):
        print("EP-{}:{}: Batch loss: {:.3f}, Acc: {:.3f}, F1: {:.3f}, "
              "TP:{:.3f}, TN:{:.3f}, FP:{:.3f}, FN:{:.3f}".format(epoch,
                                                                  batch_id,
                                                                  stats["loss"],
                                                                  stats["acc"],
                                                                  stats["f1"],
                                                                  stats["tp"] / stats["pos"],
                                                                  stats["tn"] / stats["neg"],
                                                                  stats["fp"] / stats["neg"],
                                                                  stats["fn"] / stats["pos"]))

    @staticmethod
    def print_stats(epoch, stats, split="train"):
        print("------------------------")
        print("EP-{}: {} loss: {:.3f}, Acc: {:.3f}, F1: {:.3f}, "
              "TP:{:.3f}, TN:{:.3f}, FP:{:.3f}, FN:{:.3f}".format(epoch,
                                                                  split,
                                                                  stats["loss"],
                                                                  stats["acc"],
                                                                  stats["f1"],
                                                                  stats["tp"],
                                                                  stats["tn"],
                                                                  stats["fp"],
                                                                  stats["fn"]))
        print("------------------------")

    @staticmethod
    def get_confusion_matrix(labels, classes):
        # True positive: label=1 and class=1
        tp = (classes[labels == 1] == 1).sum()
        # True negative: label=0 and class=0
        tn = (classes[labels == 0] == 0).sum()
        # False positive: label=0 and class=1
        fp = (classes[labels == 0] == 1).sum()
        # False negative: label=1 and class=0
        fn = (classes[labels == 1] == 0).sum()

        return tp, tn, fp, fn

    def save_stats(self, stats_path):
        json.dump({"train": self.train_stats, "val": self.val_stats},
                  stats_path)


class CustomCollater:
    def __init__(self, n_vars=100, random_shuffle=True, seed=1231, neg_to_pos_ratio=1):
        self.n_vars = n_vars
        self.random_shuffle = random_shuffle
        self.seed = seed
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def get_states(self, items):
        n_items = len(items)
        states = np.ones((n_items, self.n_vars)) * -1

        for i, indices in enumerate(items):
            states[i][:len(indices)] = indices

        return states

    def __call__(self, batch):
        max_pos = 0

        pids, pids_index, lids, vids, states, labels = [], [], [], [], None, []
        for item in batch:
            item["json"] = ast.literal_eval(item["json"].decode("utf-8"))

            n_pos = len(item["json"]["pos"])
            n_neg = min(len(item["json"]["neg"]), self.neg_to_pos_ratio * n_pos)

            if item["json"]["pid"] not in pids:
                pids.append(item["json"]["pid"])
            index = pids.index(item["json"]["pid"])
            pids_index.extend([index] * (n_pos + n_neg))
            lids.extend([item["json"]["lid"]] * (n_pos + n_neg))
            vids.extend([item["json"]["vid"]] * (n_pos + n_neg))
            labels.extend([1] * n_pos)
            labels.extend([0] * n_neg)

            states_pos = self.get_states(item["json"]["pos"])
            states = states_pos if states is None else np.vstack((states, states_pos))

            # For validation set keep the shuffling fixed
            if not self.random_shuffle:
                random.seed(self.seed)
            random.shuffle(item["json"]["neg"])

            states_neg = self.get_states(item["json"]["neg"][:n_neg])
            states = states_neg if states is None else np.vstack((states, states_neg))

        pids, pids_index, lids, vids = (torch.tensor(pids).int(), torch.tensor(pids_index).int(),
                                        torch.tensor(lids).int(), torch.tensor(vids).int())
        indices = torch.from_numpy(states).int()
        labels = torch.tensor(labels).long()

        return pids, pids_index, lids, vids, indices, labels


def load_inst_data(prob_name, size, split, from_pid, to_pid):
    global obj, adj
    for pid in range(from_pid, to_pid):
        data = get_instance_data(prob_name, size, split, pid)
        obj[pid] = torch.tensor(data["obj_coeffs"])
        adj[pid] = torch.from_numpy(data["adj_list"])


def train_batch(epoch, batch, model, optimizer, device, helper):
    global obj, adj
    pids, pids_index, lids, vids, indices, labels = batch
    obj_batch, adj_batch = [obj[pid] for pid in pids.tolist()], [adj[pid] for pid in pids.tolist()]
    obj_batch, adj_batch = torch.stack(obj_batch) / 100, torch.stack(adj_batch)
    # Get logits
    logits = model(obj_batch.to(device),
                   adj_batch.to(device),
                   pids_index.to(device),
                   lids.to(device),
                   vids.to(device),
                   indices.to(device))
    # Compute loss
    logits, labels = logits.reshape(-1, 2), labels.to(device).reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

    # Learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute stats
    classes = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    labels, classes = labels.cpu().numpy(), classes.cpu().numpy()
    n_pos = labels.sum()
    n_neg = labels.shape[0] - n_pos
    tp, tn, fp, fn = helper.get_confusion_matrix(labels, classes)
    acc = (tp + tn) / (n_pos + n_neg)
    f1 = tp / (tp + (0.5 * (fn + fp)))

    return {"loss": loss.item(), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "f1": f1,
            "pos": n_pos, "neg": n_neg}


def validate(epoch, dataloader, model, device, helper):
    model.eval()
    global obj, adj

    stats = helper.reset_epoch_stats()
    with torch.no_grad():
        for batch in dataloader:
            pids, pids_index, lids, vids, indices, labels = batch
            obj_batch, adj_batch = [obj[pid] for pid in pids.tolist()], [adj[pid] for pid in pids.tolist()]
            obj_batch, adj_batch = torch.stack(obj_batch) / 100, torch.stack(adj_batch)

            # Get logits
            logits = model(obj_batch.to(device),
                           adj_batch.to(device),
                           pids_index.to(device),
                           lids.to(device),
                           vids.to(device),
                           indices.to(device))
            # Compute loss
            logits, labels = logits.reshape(-1, 2), labels.to(device).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

            # Compute stats
            classes = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            labels, classes = labels.cpu().numpy(), classes.cpu().numpy()
            n_pos = labels.sum()
            n_neg = labels.shape[0] - n_pos
            tp, tn, fp, fn = helper.get_confusion_matrix(labels, classes)
            batch_stats = {"loss": loss.item(), "pos": n_pos, "neg": n_neg, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
            helper.update_running_stats(stats, batch_stats)

    model.train()

    return stats


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    ckpt_path = path.resource / "checkpoint" / "d{}-p{}-b{}-h{}-t{}-v{}".format(cfg.d_emb,
                                                                                cfg.top_k,
                                                                                cfg.n_blocks,
                                                                                cfg.n_heads,
                                                                                cfg.dataset.shard.train.end,
                                                                                cfg.dataset.shard.val.end)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    cfg.size = get_size(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = TrainingHelper()
    # Get dataset and dataloader
    load_inst_data(cfg.prob.name, cfg.size, "train", 0, 500)
    load_inst_data(cfg.prob.name, cfg.size, "val", 1000, 1100)

    dataset_root_path = str(path.dataset) + f"/{cfg.prob.name}/{cfg.size}"
    val_dataloader = helper.get_dataloader(dataset_root_path, cfg.prob.n_vars, "val", cfg.dataset.shard.val.start,
                                           cfg.dataset.shard.val.end, cfg.batch_size,
                                           num_workers=cfg.n_worker_dataloader,
                                           shardshuffle=False, random_shuffle=False)
    best_f1, best_tp = 0, 0
    model = ParetoStatePredictorMIS(encoder_type="transformer",
                                    n_node_feat=2,
                                    n_edge_type=2,
                                    d_emb=cfg.d_emb,
                                    top_k=cfg.top_k,
                                    n_blocks=cfg.n_blocks,
                                    n_heads=cfg.n_heads,
                                    bias_mha=cfg.bias_mha,
                                    dropout_mha=cfg.dropout_mha,
                                    bias_mlp=cfg.bias_mha,
                                    dropout_mlp=cfg.dropout_mlp,
                                    h2i_ratio=cfg.h2i_ratio,
                                    device=device).to(device)
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr)
    shard_start = cfg.dataset.shard.train.start

    for epoch in range(cfg.epochs):
        # Update training dataset
        if epoch % cfg.refresh_training_dataset == 0 or epoch == 0:
            shard_start = shard_start if shard_start < cfg.dataset.shard.train.end else cfg.dataset.shard.train.start
            start, end = shard_start, shard_start + cfg.shards_per_epoch - 1
            print("Training on shards: {} {}".format(start, end))

            train_dataloader = helper.get_dataloader(dataset_root_path, cfg.prob.n_vars, "train", start, end,
                                                     cfg.batch_size, num_workers=cfg.n_worker_dataloader,
                                                     shardshuffle=True, random_shuffle=True)
            shard_start += cfg.shards_per_epoch

        # Reset training stats per epoch
        stats = helper.reset_epoch_stats()

        for batch_id, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch_stats = train_batch(epoch, batch, model, optimizer, device, helper)
            batch_stats["time"] = time.time() - start_time
            helper.update_running_stats(stats, batch_stats)
            if cfg.logging > 1:
                helper.print_batch_stats(epoch, batch_id, batch_stats)

            if epoch == 0 and batch_id == 0:
                print("Validating on shards: {}..{}".format(cfg.dataset.shard.val.start,
                                                            cfg.dataset.shard.val.end))
                val_stats = validate(epoch, val_dataloader, model, device, helper)
                helper.compute_epoch_stats(val_stats)
                helper.print_stats(epoch, val_stats, split="Val")
                helper.add_epoch_stats("val", (0, val_stats))

        helper.compute_epoch_stats(stats)
        helper.print_stats(epoch, stats)
        helper.add_epoch_stats("train", (epoch, stats))

        # Validate
        if (epoch + 1) % cfg.validate_every == 0:
            print("Validating on shards: {}..{}".format(cfg.dataset.shard.val.start,
                                                        cfg.dataset.shard.val.end))
            val_stats = validate(epoch, val_dataloader, model, device, helper)
            helper.compute_epoch_stats(val_stats)
            helper.print_stats(epoch, val_stats, split="Val")
            helper.add_epoch_stats("val", (epoch, val_stats))
            if val_stats["f1"] > best_f1:
                torch.save({"epoch": epoch,
                            "best_model": model.cpu().state_dict(),
                            "optimizer": optimizer.state_dict()},
                           ckpt_path / "best_model.pt")
                best_f1 = val_stats["f1"]

        torch.save({"epoch": epoch,
                    "train": helper.train_stats,
                    "val": helper.val_stats},
                   ckpt_path / "ckpt_stats.pt")
        torch.save({"epoch": epoch,
                    "model": model.cpu().state_dict(),
                    "optimizer": optimizer.state_dict()},
                   ckpt_path / "ckpt_model.pt")


if __name__ == "__main__":
    main()
