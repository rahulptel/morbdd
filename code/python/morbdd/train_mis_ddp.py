import os
import time

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import random

from morbdd.model import ParetoStatePredictorMIS
from morbdd.train_mis import MISTrainingHelper
from morbdd.utils import Meter
from morbdd.utils.mis import get_size


def setup_ddp(cfg):
    world_size = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    # print("World size", world_size)
    n_gpus_per_node = torch.cuda.device_count()
    world_size *= n_gpus_per_node
    # print("World size", world_size)
    # The id of the node on which the current process is running
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    # print(node_id)
    # The id of the current process inside a node
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    # print("Local rank ", local_rank)
    # Unique id of the current process across processes spawned across all nodes
    global_rank = (node_id * n_gpus_per_node) + local_rank
    # print("Global rank: ", global_rank)
    # The local cuda device id we assign to the current process
    device_id = local_rank
    print("World size: {}, Rank: {}, Node: {}, Local Rank: {}".format(world_size, global_rank, node_id, local_rank))
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


def reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=None, batch_time=None):
    dist.all_reduce(losses.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(tp.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(tn.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(fp.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(fn.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(n_pos.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(n_neg.sum, dist.ReduceOp.SUM, async_op=False)
    data_time_val, batch_time_val = None, None
    if data_time is not None:
        dist.all_reduce(data_time.avg, dist.ReduceOp.AVG, async_op=False)
        data_time_val = data_time.avg
    if batch_time is not None:
        dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG, async_op=False)
        batch_time_val = batch_time.avg

    return {"loss": losses.sum, "tp": tp.sum, "tn": tn.sum, "fp": fp.sum, "fn": fn.sum,
            "n_pos": n_pos.sum, "n_neg": n_neg.sum, "batch_time": batch_time_val, "data_time": data_time_val}


def print_stats(split, stats):
    print_str = "{}:{}: F1: {:4f}, Acc: {:.4f}, Loss {:.4f}, Recall: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, "
    print_str += "Epoch Time: {:.2f}, Batch Time: {:.4f}, Data Time: {:.4f}"
    # ept, bt, dt = -1, -1, -1
    # if split == "train":
    ept, bt, dt = stats["epoch_time"], stats["batch_time"], stats["data_time"]
    print(print_str.format(stats["epoch"], split, stats["f1"], stats["acc"], stats["loss"], stats["recall"],
                           stats["precision"], stats["specificity"], ept, bt, dt))


def compute_meta_stats_and_print(epoch, split, stats_lst, stats):
    stats = {k: v.cpu().numpy() if v is not None else None for k, v in stats.items()}
    meta_stats = compute_meta_stats(stats)
    meta_stats.update({"epoch": epoch + 1})
    stats_lst.append(meta_stats)
    print_stats(split, stats_lst[-1])


def train(dataloader, model, optimizer, device, clip_grad=1.0, norm_type=2.0):
    data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
    losses = Meter('Loss')
    tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
    n_pos, n_neg = Meter('n_pos'), Meter('n_neg')

    model.train()
    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
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
        tp.update(_tp)
        tn.update(_tn)
        fp.update(_fp)
        fn.update(_fn)
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

    epoch_stats = reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=data_time, batch_time=batch_time)
    return epoch_stats


@torch.no_grad()
def validate(dataloader, model, device):
    data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
    losses = Meter('Loss')
    tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
    n_pos, n_neg = Meter('n_pos'), Meter('n_neg')

    model.eval()
    start_time = time.time()
    for batch in dataloader:
        batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.to(device).long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = compute_batch_stats(labels, preds)

        losses.update(loss.detach(), labels.size(0))
        tp.update(_tp)
        tn.update(_tn)
        fp.update(_fp)
        fn.update(_fn)
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

    epoch_stats = reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=data_time, batch_time=batch_time)

    return epoch_stats


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    # Initialize size, helper and checkpoint path
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)
    ckpt_path = helper.get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)
    print("Checkpoint path: {}".format(ckpt_path))

    # Set-up DDP and device
    rank, device_id = setup_ddp(cfg)
    master = rank == 0
    device = torch.device(device_id)
    train_stats_lst, val_stats_lst = [], []

    # Initialize train data loader with distributed sampler
    # Use pin-memory for async
    train_dataset = helper.get_dataset("train",
                                       cfg.dataset.train.from_pid,
                                       cfg.dataset.train.to_pid)
    train_sampler, train_dataloader = None, None

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
    model = DDP(model, device_ids=[device_id])
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)
    if master:
        print("Started Training...\n")

    for epoch in range(cfg.epochs):
        train_sampler, train_dataloader = get_train_sampler_and_dataloader(cfg, train_dataset, train_sampler,
                                                                           train_dataloader)
        train_sampler.set_epoch(epoch)

        if master and epoch == 0:
            print("Train loader: {}, Val loader {}".format(len(train_dataloader), len(val_dataloader)))
            print("Train samples: {}, Val samples {}".format(len(train_dataloader.dataset), len(val_dataset)))

        start_time = time.time()
        stats = train(train_dataloader, model, optimizer, device, clip_grad=cfg.clip_grad, norm_type=cfg.norm_type)
        epoch_time = torch.tensor(time.time() - start_time, dtype=torch.float32, device=device)
        dist.all_reduce(epoch_time, dist.ReduceOp.AVG, async_op=False)
        if master:
            stats["epoch_time"] = epoch_time
            compute_meta_stats_and_print(epoch, "train", train_stats_lst, stats)

        if (epoch + 1) % cfg.validate_every == 0:
            start_time = time.time()
            stats = validate(val_dataloader, model, device)
            epoch_time = torch.tensor(time.time() - start_time, dtype=torch.float32, device=device)
            dist.all_reduce(epoch_time, dist.ReduceOp.AVG, async_op=False)
            if master:
                stats["epoch_time"] = epoch_time
                compute_meta_stats_and_print(epoch, "val", val_stats_lst, stats)
                if val_stats_lst[-1]["f1"] > best_f1:
                    best_f1 = val_stats_lst[-1]["f1"]
                    val_stats_lst[-1]["is_best"] = 1
                    helper.save_model_and_opt(epoch,
                                              ckpt_path,
                                              best_model=True,
                                              model=model.module.state_dict(),
                                              optimizer=optimizer.state_dict())
                    print("* Best F1: {}".format(best_f1))
                print()

        if master and (epoch + 1) % cfg.save_every == 0:
            helper.save_model_and_opt(epoch,
                                      ckpt_path,
                                      best_model=False,
                                      model=model.module.state_dict(),
                                      optimizer=optimizer.state_dict())
            helper.save_stats(ckpt_path, train_stats=train_stats_lst, val_stats=val_stats_lst)


if __name__ == "__main__":
    main()
