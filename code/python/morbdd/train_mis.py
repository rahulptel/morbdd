import ast
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictor
from morbdd.utils import get_instance_data

random.seed(42)
rng = np.random.RandomState(42)
seeds = rng.randint(1, 10000, 1000)

obj, adj = {}, {}


class TrainingHelper:
    def __init__(self):
        self.train_stats = []
        self.val_stats = []

    def add_epoch_stats(self, split, stats):
        stat_lst = getattr(self, f"{split}_stats")
        stat_lst.append(stats)

    @staticmethod
    def get_dataloader(root_path, split, start, end, batch_size, shardshuffle=True, random_shuffle=True):
        dataset_path = root_path + f"/{split}/bdd-layer-{start}..{end}.tar"
        dataset = wds.WebDataset(dataset_path, shardshuffle=shardshuffle).shuffle(5000)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=CustomCollater(random_shuffle=random_shuffle))

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
        stats["loss"] = stats["loss"] / stats["items"]
        stats["acc"] = ((stats["tp"] + stats["fp"]) /
                        stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"])
        stats["f1"] = stats["tp"] / (stats["tp"] + (0.5 * (stats["fn"] + stats["fp"])))
        stats["tp"] = stats["tp"] / stats["items"]
        stats["fp"] = stats["fp"] / stats["items"]
        stats["tn"] = stats["tn"] / stats["items"]
        stats["fn"] = stats["fn"] / stats["items"]

    @staticmethod
    def print_batch_stats(epoch, batch_id, stats):
        print("\t\tEP-{}:{}: Batch loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}, "
              "TP:{:.4f}, TN:{:.4f}, FP:{:.4f}, FN:{:.4f}".format(epoch,
                                                                  batch_id,
                                                                  stats["loss"],
                                                                  stats["acc"],
                                                                  stats["f1"],
                                                                  stats["tp"] / stats["pos"],
                                                                  stats["tn"] / stats["neg"],
                                                                  stats["fp"] / stats["neg"],
                                                                  stats["fn"] / stats["pos"]))

    @staticmethod
    def print_stats(epoch, batch_id, stats):
        print("EP-{}: Batch loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}, "
              "TP:{:.4f}, TN:{:.4f}, FP:{:.4f}, FN:{:.4f}".format(epoch,
                                                                  stats["loss"],
                                                                  stats["acc"],
                                                                  stats["f1"],
                                                                  stats["tp"],
                                                                  stats["tn"],
                                                                  stats["fp"],
                                                                  stats["fn"]))

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


class CustomCollater:
    def __init__(self, random_shuffle=True, seed=1231):
        self.random_shuffle = random_shuffle
        self.seed = seed

    def __call__(self, batch):
        max_pos = 0
        for item in batch:
            item["json"] = ast.literal_eval(item["json"].decode("utf-8"))
            if len(item["json"]["pos"]) > max_pos:
                max_pos = len(item["json"]["pos"])

        pids, lids, vids = [], [], []
        indices = np.ones((len(batch), 2 * max_pos, 100)) * -1
        weights = np.zeros((len(batch), 2 * max_pos))
        labels = np.zeros((len(batch), 2 * max_pos))
        for ibatch, item in enumerate(batch):
            pids.append(item["json"]["pid"])
            lids.append(item["json"]["lid"])
            vids.append(item["json"]["vid"])
            i = 0
            for ipos, pos_ind in enumerate(item["json"]["pos"]):
                indices[ibatch][i][:len(pos_ind)] = pos_ind
                weights[ibatch][i] = 1
                labels[ibatch][i] = 1
                i += 1

            # For validation set keep the shuffling fixed
            if not self.random_shuffle:
                random.seed(self.seed)
            random.shuffle(item["json"]["neg"])

            neg_max = min(len(item["json"]["pos"]), len(item["json"]["neg"]))
            for ineg, neg_ind in enumerate(item["json"]["neg"][: neg_max]):
                indices[ibatch][i][:len(neg_ind)] = neg_ind
                weights[ibatch][i] = 1
                i += 1

        pids, lids, vids = torch.tensor(pids).int(), torch.tensor(lids).int(), torch.tensor(vids).int()
        indices = torch.from_numpy(indices).int()
        weights = torch.from_numpy(weights).float()
        labels = torch.from_numpy(labels).long()

        return pids, lids, vids, indices, weights, labels


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def load_inst_data(prob_name, size, split, from_pid, to_pid):
    global obj, adj
    for pid in range(from_pid, to_pid):
        data = get_instance_data(prob_name, size, split, pid)
        obj[pid] = torch.tensor(data["obj_coeffs"])
        adj[pid] = torch.from_numpy(data["adj_list"])


def train_batch(epoch, batch, model, optimizer, device):
    global obj, adj
    pids, lids, vids, indices, weights, labels = batch
    obj_batch, adj_batch = [obj[pid] for pid in pids.tolist()], [adj[pid] for pid in pids.tolist()]
    obj_batch, adj_batch = torch.stack(obj_batch), torch.stack(adj_batch).int()
    obj_batch = obj_batch / 100

    # Get logits
    logits = model(obj_batch.to(device), adj_batch.to(device), lids.to(device), vids.to(device), indices.to(device))
    # Compute loss
    logits, labels = logits.reshape(-1, 2), labels.to(device).reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction='none')
    weights = weights.to(device)
    scaled_loss = (weights.reshape(-1) * loss).sum() / weights.sum()
    # Learn
    optimizer.zero_grad()
    scaled_loss.backward()
    optimizer.step()

    classes = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    labels, classes = labels[weights.reshape(-1) == 1], classes[weights.reshape(-1) == 1]
    labels, classes = labels.cpu().numpy(), classes.cpu().numpy()
    n_pos = labels.sum()
    n_neg = labels.shape[0] - n_pos

    tp, tn, fp, fn = get_confusion_matrix(labels, classes)

    return {"loss": scaled_loss.item(), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "pos": n_pos, "neg": n_neg}


def validate(epoch, dataloader, model, device):
    model.eval()
    global obj, adj

    with torch.no_grad():
        epoch_loss, epoch_items, epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0, 0, 0
        epoch_n_pos, epoch_n_neg = 0, 0

        for batch in dataloader:
            pids, lids, vids, indices, weights, labels = batch
            obj_batch, adj_batch = [obj[pid] for pid in pids.tolist()], [adj[pid] for pid in pids.tolist()]
            obj_batch, adj_batch = torch.stack(obj_batch), torch.stack(adj_batch).int()
            obj_batch = obj_batch / 100

            # Get logits
            logits = model(obj_batch.to(device), adj_batch.to(device), lids.to(device), vids.to(device),
                           indices.to(device))
            # Compute loss
            logits, labels = logits.reshape(-1, 2), labels.to(device).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction='none')
            weights = weights.to(device)
            scaled_loss = (weights.reshape(-1) * loss).sum() / weights.sum()

            classes = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            labels, classes = labels[weights.reshape(-1) == 1], classes[weights.reshape(-1) == 1]
            labels, classes = labels.cpu().numpy(), classes.cpu().numpy()
            n_pos = labels.sum()
            n_neg = labels.shape[0] - n_pos

            tp, tn, fp, fn = get_confusion_matrix(labels, classes)

            epoch_loss += (scaled_loss.item() * (n_pos + n_neg))
            epoch_items += (n_pos + n_neg)
            epoch_tp += tp
            epoch_fp += fp
            epoch_tn += tn
            epoch_fn += fn
            epoch_n_pos += n_pos
            epoch_n_neg += n_neg

            print("\t\t Val Batch loss: {:.4f} TP:{:.4f} TN:{:.4f} FP:{:.4f} FN:{:.4f}".format(
                scaled_loss.item(),
                tp / n_pos,
                tn / n_neg,
                fp / n_neg,
                fn / n_pos))

        print("\t{}: Val epoch loss: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(epoch,
                                                                                epoch_loss / epoch_items,
                                                                                epoch_tp / epoch_n_pos,
                                                                                epoch_tn / epoch_n_neg,
                                                                                epoch_fp / epoch_n_neg,
                                                                                epoch_fn / epoch_n_pos))

    model.train()


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = TrainingHelper()
    # Get dataset and dataloader
    load_inst_data(cfg.prob.name, cfg.size, "train", cfg.dataset.pid.train.start, cfg.dataset.pid.train.end)
    load_inst_data(cfg.prob.name, cfg.size, "val", cfg.dataset.pid.val.start, cfg.dataset.pid.val.end)

    dataset_root_path = str(path.dataset) + f"/{cfg.prob.name}/{cfg.size}"
    val_dataloader = helper.get_dataloader(dataset_root_path, "val", 0, 9, 128,
                                           shardshuffle=False, random_shuffle=False)

    model = ParetoStatePredictor(d_emb=64,
                                 top_k=5,
                                 n_blocks=2,
                                 n_heads=8,
                                 bias_mha=False,
                                 dropout_mha=0,
                                 bias_mlp=True,
                                 dropout_mlp=0.1,
                                 h2i_ratio=2,
                                 node_residual=True,
                                 edge_residual=True).to(device)
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr)
    batch_counter, inst_start = 0, cfg.dataset.pid.train.start

    for epoch in range(cfg.epochs):
        if epoch % cfg.refresh_training_dataset or epoch == 0:
            start, end = inst_start, inst_start + cfg.n_inst_per_group
            train_dataloader = helper.get_dataloader(dataset_root_path, "train", start, end, cfg.batch_size,
                                                     shardshuffle=True, random_shuffle=True)
            inst_start = inst_start + cfg.n_inst_per_group if inst_start < cfg.dataset.pid.train.end else 0

        stats = helper.reset_epoch_stats()
        for batch_id, batch in enumerate(train_dataloader):
            batch_stats = train_batch(epoch, batch, model, optimizer, device)
            helper.update_epoch_stats(stats, batch_stats)
            if cfg.logging > 1:
                helper.print_stats(epoch, batch_id, batch_stats)

            if epoch == 0 and batch_id == 1:
                val_stats = validate(epoch, val_dataloader, model, device)
                helper.add_epoch_stats("val", val_stats)

        helper.print_stats(epoch, -1, stats)
        if (epoch + 1) % cfg.validate_every == 0:
            val_stats = validate(epoch, val_dataloader, model, device)
            helper.add_epoch_stats("val", val_stats)


if __name__ == "__main__":
    main()
