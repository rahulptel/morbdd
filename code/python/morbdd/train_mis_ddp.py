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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from morbdd.train_mis import MISBDDNodeDataset
from morbdd.train_mis import MISTrainingHelper
import torch.distributed as dist
from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def setup_ddp(cfg):
    world_size = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    print("World size", world_size)
    n_gpus_per_node = torch.cuda.device_count()
    world_size *= n_gpus_per_node
    print("World size", world_size)
    # The id of the node on which the current process is running
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    print(node_id)
    # The id of the current process inside a node
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("Local rank ", local_rank)
    # Unique id of the current process across processes spawned across all nodes
    global_rank = (node_id * n_gpus_per_node) + local_rank
    print("Global rank: ", global_rank)
    # The local cuda device id we assign to the current process
    device_id = local_rank
    # Initialize process group and initiate communications between all processes
    # running on all nodes
    print("From Rank: {}, ==> Initializing Process Group...".format(global_rank))
    # init the process group
    init_process_group(backend=cfg.dist_backend,
                       init_method=cfg.init_method,
                       world_size=world_size,
                       rank=global_rank)
    print("Process group ready!")
    print("From Rank: {}, ==> Making model...".format(global_rank))

    return global_rank, device_id


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


def compute_meta_stats(*stats):
    losses, tp, tn, fp, fn, n_pos, n_neg = stats
    losses, tp, tn, fp, fn, n_pos, n_neg = (losses.numpy(), tp.numpy(), tn.numpy(), fp.numpy(), fn.numpy(),
                                            n_pos.numpy(), n_neg.numpy())
    losses = losses / (n_pos + n_neg)
    acc = ((tp + tn) / (tp + fp + tn + fn))
    f1 = tp / (tp + (0.5 * (fn + fp)))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    return {"losses": losses, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "precision": precision,
            "recall": recall, "specificity": specificity}


def train(dataloader, model, optimizer, device, clip_grad=1.0, norm_type=2.0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    tp = AverageMeter('TP', ':6.3f')
    fp = AverageMeter('FP', ':6.3f')
    tn = AverageMeter('TN', ':6.3f')
    fn = AverageMeter('FN', ':6.3f')
    n_pos = AverageMeter('n_pos')
    n_neg = AverageMeter('n_neg')

    model.train()
    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.to(device).long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Backprop with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=norm_type)
        optimizer.step()

        # Compute stats
        preds = torch.argmax(F.softmax(logits.detach(), dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = compute_batch_stats(labels, preds)
        losses.update(loss.detach(), labels.size(0))
        tp.update(_tp, labels.size(0))
        tn.update(_tn, labels.size(0))
        fp.update(_fp, labels.size(0))
        fn.update(_fn, labels.size(0))
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

    losses = dist.all_reduce(losses.sum, dist.ReduceOp.SUM)
    tp = dist.all_reduce(tp, dist.ReduceOp.SUM)
    tn = dist.all_reduce(tn, dist.ReduceOp.SUM)
    fp = dist.all_reduce(fp, dist.ReduceOp.SUM)
    fn = dist.all_reduce(fn, dist.ReduceOp.SUM)
    n_pos = dist.all_reduce(n_pos, dist.ReduceOp.SUM)
    n_neg = dist.all_reduce(n_neg, dist.ReduceOp.SUM)

    return losses, tp, tn, fp, fn, n_pos, n_neg


@torch.no_grad()
def validate(dataloader, model, device):
    losses = AverageMeter('Loss', ':.4e')
    tp = AverageMeter('TP', ':6.3f')
    fp = AverageMeter('FP', ':6.3f')
    tn = AverageMeter('TN', ':6.3f')
    fn = AverageMeter('FN', ':6.3f')
    n_pos = AverageMeter('n_pos')
    n_neg = AverageMeter('n_neg')

    model.eval()
    for batch in dataloader:
        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.to(device).long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = compute_batch_stats(labels, preds)
        losses.update(loss.detach(), labels.size(0))
        tp.update(_tp, labels.size(0))
        tn.update(_tn, labels.size(0))
        fp.update(_fp, labels.size(0))
        fn.update(_fn, labels.size(0))
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)

    losses = dist.all_reduce(losses.sum, dist.ReduceOp.SUM)
    tp = dist.all_reduce(tp, dist.ReduceOp.SUM)
    tn = dist.all_reduce(tn, dist.ReduceOp.SUM)
    fp = dist.all_reduce(fp, dist.ReduceOp.SUM)
    fn = dist.all_reduce(fn, dist.ReduceOp.SUM)
    n_pos = dist.all_reduce(n_pos, dist.ReduceOp.SUM)
    n_neg = dist.all_reduce(n_neg, dist.ReduceOp.SUM)

    return losses, tp, tn, fp, fn, n_pos, n_neg


def print_stats(epoch, split, stats):
    pass


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    # Initialize size, helper and checkpoint path
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)
    ckpt_path = get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    # Set-up DDP and device
    rank, device_id = setup_ddp(cfg)
    master = rank == 0
    device = torch.device(device_id)
    if master:
        train_stats_lst, val_stats_lst = [], []

    # Initialize train data loader with distributed sampler
    # Use pin-memory for async
    train_dataset = helper.get_dataset("train",
                                       cfg.dataset.train.from_pid,
                                       cfg.dataset.train.to_pid)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  num_workers=cfg.n_worker_dataloader,
                                  pin_memory=True)
    # Initialize val data loader with distributed sampler
    val_dataset = helper.get_dataset("val",
                                     cfg.dataset.val.from_pid,
                                     cfg.dataset.val.to_pid)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                sampler=val_sampler,
                                num_workers=cfg.n_worker_dataloader,
                                pin_memory=True)
    if master:
        print("Train samples: {}, Val samples {}".format(len(train_dataset),
                                                         len(val_dataset)))
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
    model = DDP(model, device_ids=[device_id])
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        stats = train(train_dataloader, model, optimizer, device, clip_grad=cfg.clip_grad, norm_type=cfg.norm_type)
        if master:
            meta_stats = compute_meta_stats(stats)
            meta_stats.update({"epoch": epoch + 1})
            train_stats_lst.append(meta_stats)
            print_stats(epoch, "train", train_stats_lst[-1])

        best_model = False
        if (epoch + 1) % cfg.validate_every == 0:
            stats = validate(val_dataloader, model, device)
            if master:
                meta_stats = compute_meta_stats(stats)
                meta_stats.update({"epoch": epoch + 1})
                val_stats_lst.append(meta_stats)
                if val_stats_lst[-1]["f1"] > best_f1:
                    best_f1 = val_stats_lst[-1]["f1"]
                    best_model = True
                    print("***{} Best f1: {}".format(epoch + 1, best_f1))

        # if master and (epoch + 1) % cfg.save_every == 0:
        #     helper.save(epoch,
        #                 ckpt_path,
        #                 best_model=best_model,
        #                 model=model.module.state_dict(),
        #                 optimizer=optimizer.state_dict())


if __name__ == "__main__":
    main()
