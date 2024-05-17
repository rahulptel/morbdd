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

obj, adj = {}, {}


class MISInstanceDataset(Dataset):
    def __init__(self, size, split, from_pid, to_pid, device, top_k=5):
        super(MISInstanceDataset, self).__init__()
        self.obj, self.adj = [], []
        for pid in range(from_pid, to_pid):
            data = get_instance_data(size, split, pid)
            self.obj.append(torch.tensor(data["obj_coeffs"]))
            self.adj.append(torch.from_numpy(data["adj_list"]))
        self.obj = torch.stack(self.obj)
        self.adj = torch.stack(self.adj)
        self.pos_enc = self.precompute_pos_enc(top_k)

    def __getitem__(self, item):
        return self.obj[item], self.adj[item], self.pos_enc[item]

    def __len__(self):
        return self.adj.shape[0]

    def precompute_pos_enc(self, top_k):
        p = None
        if top_k > 0:
            # Calculate position encoding
            U, S, Vh = torch.linalg.svd(self.adj)
            U = U[:, :, :top_k]
            S = (torch.diag_embed(S)[:, :top_k, :top_k]) ** (1 / 2)
            Vh = Vh[:, :top_k, :]

            L, R = U @ S, S @ Vh
            R = R.permute(0, 2, 1)
            p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)

        return p


class MISBDDNodeDataset(Dataset):
    def __init__(self, root_path, split, device):
        super(MISBDDNodeDataset, self).__init__()
        self.nodes = torch.from_numpy(np.load(root_path / f"{split}.npy"))

    def __getitem__(self, item):
        return (self.nodes[item, 0],
                self.nodes[item, 1],
                self.nodes[item, 2],
                self.nodes[item, 3:103],
                self.nodes[item, 103])

    def __len__(self):
        return self.nodes.shape[0]


class MISTrainingHelper(TrainingHelper):
    def __init__(self, cfg):
        super(MISTrainingHelper, self).__init__()
        self.cfg = cfg

    def get_inst_dataset(self, split, from_pid, to_pid, device=None):
        inst_dataset = MISInstanceDataset(self.cfg.size, split, from_pid, to_pid, device=None)
        return inst_dataset

    @staticmethod
    def get_dataloader_v1(root_path, n_vars, split, start, end, batch_size, num_workers=0, shardshuffle=True,
                          random_shuffle=True, device=None):
        dataset_path = root_path + f"/{split}/" + "bdd-layer-{" + f"{start}..{end}" + "}.tar"
        dataset = wds.WebDataset(dataset_path, shardshuffle=shardshuffle).shuffle(5000)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                collate_fn=CustomCollater(n_vars=n_vars, random_shuffle=random_shuffle))

        return dataloader

    def get_dataloader_v2(self, root_path, size, split, from_pid, to_pid, batch_size, num_workers=1, device=None):
        node_dataset = MISBDDNodeDataset(root_path, split, device=device)
        dataloader = DataLoader(node_dataset, batch_size=batch_size, num_workers=num_workers)

        return dataloader

    def get_dataloader(self, version="v2", root_path=None, n_vars=3, split="train", start=0, end=1000, batch_size=64,
                       num_workers=0, shardshuffle=True, random_shuffle=True, device=torch.device('cpu')):
        if version == "v2":
            return self.get_dataloader_v1(root_path, n_vars, split, start, end, batch_size, num_workers=num_workers,
                                          shardshuffle=shardshuffle, random_shuffle=random_shuffle)

        elif version == "v2":
            return self.get_dataloader_v2(root_path, split, batch_size, num_workers=num_workers)


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
    preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
    batch_stats = helper.compute_batch_stats(labels, preds)
    batch_stats["loss"] = loss.item()

    return batch_stats


def validate(epoch, dataloader, model, device, helper):
    global obj, adj
    stats = helper.reset_epoch_stats()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            start = time.time()
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
    helper = MISTrainingHelper()
    ckpt_path = get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    dataset_root_path = str(path.dataset) + f"/{cfg.prob.name}/{cfg.size}"
    val_dataloader = helper.get_dataloader(version=cfg.dataset_version,
                                           root_path=dataset_root_path,
                                           n_vars=cfg.prob.n_vars,
                                           split="val",
                                           start=cfg.dataset.shard.val.start,
                                           end=cfg.dataset.shard.val.end,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.n_worker_dataloader,
                                           shardshuffle=False,
                                           random_shuffle=False)
    best_recall = 0
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
    shard_start = cfg.dataset.shard.train.start

    for epoch in range(cfg.epochs):
        # # Update training dataset
        # if epoch % cfg.refresh_training_dataset == 0 or epoch == 0:
        #     shard_start = shard_start if shard_start < cfg.dataset.shard.train.end else cfg.dataset.shard.train.start
        #     start, end = shard_start, shard_start + cfg.shards_per_epoch - 1
        #     print("Training on shards: {}..{}".format(start, end))
        #
        #     train_dataloader = helper.get_dataloader(dataset_root_path, cfg.prob.n_vars, "train", start, end,
        #                                              cfg.batch_size, num_workers=cfg.n_worker_dataloader,
        #                                              shardshuffle=True, random_shuffle=True)
        #     shard_start += cfg.shards_per_epoch

        # Reset training stats per epoch
        stats = helper.reset_epoch_stats()

        for batch_id, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch_stats = train_batch(epoch, batch, model, optimizer, device, helper)
            batch_stats["time"] = time.time() - start_time
            helper.update_running_stats(stats, batch_stats)
            if cfg.logging > 1:
                helper.print_batch_stats(epoch, batch_id, batch_stats)

            # if epoch == 0 and batch_id == 0:
            #     print("Validating on shards: {}..{}".format(cfg.dataset.shard.val.start,
            #                                                 cfg.dataset.shard.val.end))
            #     val_stats = validate(epoch, val_dataloader, model, device, helper)
            #     helper.add_epoch_stats("val", (0, val_stats))

        helper.compute_epoch_stats(stats)
        helper.print_stats(epoch, stats)
        helper.add_epoch_stats("train", (epoch, stats))
        helper.save(epoch,
                    ckpt_path,
                    stats=True,
                    best_model=False,
                    model={k: v.cpu() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict())

        # Validate
        if (epoch + 1) % cfg.validate_every == 0:
            print("Validating on shards: {}..{}".format(cfg.dataset.shard.val.start,
                                                        cfg.dataset.shard.val.end))
            val_stats = validate(epoch, val_dataloader, model, device, helper)
            helper.add_epoch_stats("val", (epoch, val_stats))

            if val_stats["recall"] > best_recall:
                print("***{} Best recall: {}".format(epoch, val_stats["recall"]))
                helper.save(epoch,
                            ckpt_path,
                            stats=False,
                            best_model=True,
                            model={k: v.cpu() for k, v in model.state_dict().items()},
                            optimizer=optimizer.state_dict())
                best_recall = val_stats["recall"]


if __name__ == "__main__":
    main()
