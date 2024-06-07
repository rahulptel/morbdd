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
from torch.utils.data import DistributedSampler

from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import Meter
from morbdd.utils.mis import MISTrainingHelper
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


def reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=None, batch_time=None, multi_gpu=False):
    if multi_gpu:
        dist.all_reduce(losses.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(tp.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(tn.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(fp.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(fn.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(n_pos.sum, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(n_neg.sum, dist.ReduceOp.SUM, async_op=False)
        if data_time is not None:
            dist.all_reduce(data_time.avg, dist.ReduceOp.AVG, async_op=False)
        if batch_time is not None:
            dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG, async_op=False)

    stats = {"loss": losses.sum, "tp": tp.sum, "tn": tn.sum, "fp": fp.sum, "fn": fn.sum,
             "n_pos": n_pos.sum, "n_neg": n_neg.sum, "batch_time": batch_time.avg, "data_time": data_time.avg}
    stats = {k: v.cpu().numpy() for k, v in stats.items()}

    return stats


def set_epoch_time(cfg, epoch_time, stats, device):
    if cfg.multi_gpu:
        epoch_time = torch.tensor(epoch_time, dtype=torch.float32, device=device)
        dist.all_reduce(epoch_time, dist.ReduceOp.AVG, async_op=False)
        stats["epoch_time"] = epoch_time.cpu()
    else:
        stats["epoch_time"] = epoch_time


@torch.no_grad()
def validate(dataloader, model, device, helper, pin_memory=True, multi_gpu=False):
    data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
    losses = Meter('Loss')
    tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
    n_pos, n_neg = Meter('n_pos'), Meter('n_neg')

    model.eval()
    start_time = time.time()
    for batch in dataloader:
        if pin_memory:
            batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        # Get logits
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = helper.compute_batch_stats(labels, preds.detach())
        losses.update(loss.detach(), labels.shape[0])
        tp.update(_tp)
        tn.update(_tn)
        fp.update(_fp)
        fn.update(_fn)
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

    epoch_stats = reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=data_time, batch_time=batch_time,
                         multi_gpu=multi_gpu)
    return epoch_stats


def train(dataloader, model, optimizer, device, helper, clip_grad=1.0, norm_type=2.0, pin_memory=True, multi_gpu=False):
    data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
    losses = Meter('Loss')
    tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
    n_pos, n_neg = Meter('n_pos'), Meter('n_neg')

    model.train()
    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        if pin_memory:
            batch = [item.to(device, non_blocking=True) for item in batch]
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

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
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = helper.compute_batch_stats(labels, preds.detach())

        losses.update(loss.detach(), labels.shape[0])
        tp.update(_tp)
        tn.update(_tn)
        fp.update(_fp)
        fn.update(_fn)
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

    epoch_stats = reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time=data_time, batch_time=batch_time,
                         multi_gpu=multi_gpu)
    return epoch_stats


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)

    # Set-up device
    device_str, pin_memory, master, device_id = "cpu", False, True, 0
    if cfg.multi_gpu:
        rank, device_id = setup_ddp(cfg)
        master = rank == 0
        device_str = f"cuda:{device_id}"
    elif torch.cuda.is_available():
        device_str = "cuda"
        pin_memory = True
    device = torch.device(device_str)
    print("Device :", device)

    # Initialize dataloaders
    print("N worker dataloader: ", cfg.n_worker_dataloader)
    train_dataset = helper.get_dataset("train",
                                       cfg.dataset.train.from_pid,
                                       cfg.dataset.train.to_pid)
    train_sampler, train_dataloader = None, None

    val_dataset = helper.get_val_dataset(cfg, train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if cfg.multi_gpu else None
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                sampler=val_sampler,
                                shuffle=False,
                                num_workers=cfg.n_worker_dataloader,
                                pin_memory=pin_memory)

    # Initialize model and optimizer
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
                                    h2i_ratio=cfg.h2i_ratio)
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)
    start_epoch, end_epoch, best_f1 = 0, cfg.epochs, 0

    # Load model if restarting
    ckpt_path = helper.get_checkpoint_path(cfg)
    ckpt_path.mkdir(exist_ok=True, parents=True)
    print("Checkpoint path: {}".format(ckpt_path))
    if cfg.training_from == "last_checkpoint":
        ckpt = torch.load(ckpt_path / "model.pt", map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["opt_dict"])

        stats = torch.load(ckpt_path / "stats.pt", map_location=torch.device("cpu"))
        for v in stats["val"]:
            if v["f1"] < best_f1:
                best_f1 = v["f1"]

        start_epoch = int(ckpt["epoch"])

    model.to(device)
    if cfg.multi_gpu:
        model = DDP(model, device_ids=[device_id])

    # Reset training stats per epoch
    for epoch in range(start_epoch, end_epoch):
        _dataset = helper.get_train_dataset(cfg, train_dataset)
        train_sampler, train_dataloader = helper.get_train_sampler_and_dataloader(cfg, _dataset, train_sampler,
                                                                                  train_dataloader,
                                                                                  pin_memory=pin_memory)
        if cfg.multi_gpu:
            train_sampler.set_epoch(epoch)
        if master:
            print("Train samples: {}, Val samples {}".format(len(train_dataset), len(val_dataset)))
            print("Train loader: {}, Val loader {}".format(len(train_dataloader), len(val_dataloader)))

        # Train
        start_time = time.time()
        stats = train(train_dataloader, model, optimizer, device, helper, clip_grad=cfg.clip_grad,
                      norm_type=cfg.norm_type, pin_memory=pin_memory, multi_gpu=cfg.multi_gpu)
        epoch_time = time.time() - start_time
        set_epoch_time(cfg, epoch_time, stats, device)
        if master:
            stats["epoch"] = epoch + 1
            helper.compute_meta_stats_and_print("train", stats)

        # Validate
        if (epoch + 1) % cfg.validate_every == 0:
            start_time = time.time()
            stats = validate(val_dataloader, model, device, helper, pin_memory=pin_memory, multi_gpu=cfg.multi_gpu)
            epoch_time = time.time() - start_time
            set_epoch_time(cfg, epoch_time, stats, device)
            if master:
                stats["epoch"] = epoch + 1
                helper.compute_meta_stats_and_print("val", stats)
                if helper.val_stats[-1]["f1"] > best_f1:
                    best_f1 = helper.val_stats[-1]["f1"]
                    helper.save_model_and_opt(epoch, ckpt_path, best_model=True,
                                              model=(model.module.state_dict() if cfg.multi_gpu else
                                                     model.state_dict()),
                                              optimizer=optimizer.state_dict())
                    helper.save_stats(ckpt_path)
                    print("* Best F1: {}".format(best_f1))

        if master and (epoch + 1) % cfg.save_every == 0:
            helper.save_model_and_opt(epoch, ckpt_path, best_model=True,
                                      model=(model.module.state_dict() if cfg.multi_gpu else
                                             model.state_dict()),
                                      optimizer=optimizer.state_dict())
            helper.save_stats(ckpt_path)

        if master:
            print()


if __name__ == "__main__":
    main()
