import time

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import Meter
from morbdd.utils import get_device
from morbdd.utils import set_seed
from morbdd.utils.mis import MISTrainingHelper
from morbdd.utils.mis import get_size
from morbdd.utils import dict2cpu


def reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time):
    dist.all_reduce(losses.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(tp.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(tn.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(fp.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(fn.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(n_pos.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(n_neg.sum, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(data_time.avg, dist.ReduceOp.AVG, async_op=False)
    dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG, async_op=False)


def reduce_epoch_time(epoch_time, device):
    epoch_time = torch.tensor(epoch_time, dtype=torch.float32, device=device)
    dist.all_reduce(epoch_time, dist.ReduceOp.AVG, async_op=False)
    return epoch_time


def stats2dict(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time):
    return {"loss": losses.sum, "tp": tp.sum, "tn": tn.sum, "fp": fp.sum, "fn": fn.sum,
            "n_pos": n_pos.sum, "n_neg": n_neg.sum, "batch_time": batch_time.avg, "data_time": data_time.avg}


def print_validation_machine(master, val_sampler, device_str, distributed):
    if master:
        if val_sampler is None:
            if "cuda" in device_str:
                if distributed:
                    print("Multi-GPU: Validation on master GPU")
                else:
                    print("Validation on GPU")
            else:
                print("Validation on CPU")
        else:
            print("Multi-GPU: Validation across GPUs")


@torch.no_grad()
def validate(dataloader, model, device, helper, pin_memory=True, master=False, distributed=False,
             validate_on_master=True, world_size=4):
    epoch_stats = None
    if (not distributed) or (distributed and validate_on_master and master) or (distributed and not validate_on_master):
        data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
        losses = Meter('Loss')
        tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
        n_pos, n_neg = Meter('n_pos'), Meter('n_neg')
        result = torch.empty((6, 0))

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
            scores = F.softmax(logits, dim=-1)
            preds = torch.argmax(scores, dim=-1)
            _tp, _tn, _fp, _fn, _n_pos, _n_neg = helper.compute_batch_stats(labels, preds.detach())
            losses.update(loss.detach(), labels.shape[0])
            tp.update(_tp)
            tn.update(_tn)
            fp.update(_fp)
            fn.update(_fn)
            n_pos.update(_n_pos)
            n_neg.update(_n_neg)

            result = torch.cat((result,
                                torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))), dim=1)
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            start_time = time.time()

        if distributed and not validate_on_master:
            reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time)
            result_lst = [result for _ in range(world_size)]
            dist.gather(result, gather_list=result_lst, dst=0)
            result = torch.cat(result_lst)

        epoch_stats = stats2dict(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time)
        epoch_stats["lgt0"] = torch.histogram(result[:, 2], 10)
        epoch_stats["lgt1"] = torch.histogram(result[:, 3], 10)
        epoch_stats["score0"] = torch.histogram(result[result[:, 0] == 0][:, -2], 10)
        epoch_stats["score1"] = torch.histogram(result[result[:, 0] == 1][:, -2], 10)

    return epoch_stats


def train(dataloader, model, optimizer, device, helper, clip_grad=1.0, norm_type=2.0, pin_memory=True,
          distributed=False, world_size=4):
    data_time, batch_time = Meter('DataTime'), Meter('BatchTime')
    losses = Meter('Loss')
    tp, fp, tn, fn = Meter('TP'), Meter('FP'), Meter('TN'), Meter('FN')
    n_pos, n_neg = Meter('n_pos'), Meter('n_neg')
    result = torch.empty((6, 0))

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
        logits = logits.detach()
        scores = F.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(scores, dim=-1)
        _tp, _tn, _fp, _fn, _n_pos, _n_neg = helper.compute_batch_stats(labels, preds.detach())

        losses.update(loss.detach(), labels.shape[0])
        tp.update(_tp)
        tn.update(_tn)
        fp.update(_fp)
        fn.update(_fn)
        n_pos.update(_n_pos)
        n_neg.update(_n_neg)
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        result = torch.cat((result,
                            torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))), dim=1)

        start_time = time.time()

    result = result.T
    if distributed:
        reduce(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time)
        result_lst = [result for _ in range(world_size)]
        dist.gather(result, gather_list=result_lst, dst=0)
        result = torch.cat(result_lst)

    epoch_stats = stats2dict(losses, tp, tn, fp, fn, n_pos, n_neg, data_time, batch_time)
    epoch_stats["lgt0"] = torch.histogram(result[:, 2], 10)
    epoch_stats["lgt1"] = torch.histogram(result[:, 3], 10)
    epoch_stats["score0"] = torch.histogram(result[result[:, 0] == 0][:, -2], 10)
    epoch_stats["score1"] = torch.histogram(result[result[:, 0] == 1][:, -2], 10)

    return epoch_stats


def training_loop(cfg, master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader, val_dataset,
                  val_dataloader, model, optimizer, helper, device, pin_memory, ckpt_path):
    best_f1 = 0
    for epoch in range(start_epoch, end_epoch):
        # Train on the entire dataset or a subset depending on `dataset.train.frac_per_epoch`
        _dataset = helper.get_train_dataset(cfg, train_dataset)
        train_sampler, train_dataloader = helper.get_train_sampler_and_dataloader(cfg, _dataset, train_sampler,
                                                                                  train_dataloader,
                                                                                  pin_memory=pin_memory)
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        if master:
            print("Train samples: {}, Val samples {}".format(len(train_dataset), len(val_dataset)))
            print("Train loader: {}, Val loader {}".format(len(train_dataloader), len(val_dataloader)))

        # Train
        start_time = time.time()
        stats = train(train_dataloader, model, optimizer, device, helper, clip_grad=cfg.clip_grad,
                      norm_type=cfg.norm_type, pin_memory=pin_memory, distributed=cfg.distributed)
        epoch_time = time.time() - start_time
        epoch_time = reduce_epoch_time(epoch_time, device) if cfg.distributed else epoch_time

        if master:
            stats = dict2cpu(stats) if "cpu" not in str(device) else stats
            epoch_time = float(epoch_time.cpu().numpy()) if cfg.distributed else epoch_time
            stats.update({"epoch_time": epoch_time, "epoch": epoch + 1})
            helper.compute_meta_stats_and_print("train", stats)

        # Validate
        if (epoch + 1) % cfg.validate_every == 0:
            start_time = time.time()
            stats = {}
            for split in cfg.validate_on_split:
                dataloader = val_dataloader if split == "val" else train_dataloader
                new_stats = validate(dataloader, model, device, helper, pin_memory=pin_memory, master=master,
                                     distributed=cfg.distributed, validate_on_master=cfg.validate_on_master)
                if new_stats is not None:
                    new_stats = {f'tr_' + k: v for k, v in new_stats.items()} if split == "train" else new_stats
                    stats.update(new_stats)

            epoch_time = time.time() - start_time
            epoch_time = reduce_epoch_time(epoch_time, device) if cfg.distributed and not cfg.validate_on_master \
                else epoch_time

            if master:
                stats = dict2cpu(stats) if "cpu" not in str(device) else stats
                epoch_time = float(epoch_time.cpu().numpy()) if cfg.distributed and not cfg.validate_on_master \
                    else epoch_time
                stats.update({"epoch_time": epoch_time, "epoch": epoch + 1})
                helper.compute_meta_stats_and_print("val", stats)
                if helper.val_stats[-1]["f1"] > best_f1:
                    best_f1 = helper.val_stats[-1]["f1"]
                    helper.save_model_and_opt(epoch, ckpt_path, best_model=True,
                                              model=(model.module.state_dict() if cfg.distributed else
                                                     model.state_dict()),
                                              optimizer=optimizer.state_dict())
                    helper.save_stats(ckpt_path)
                    print("* Best F1: {}".format(best_f1))

        if master and (epoch + 1) % cfg.save_every == 0:
            helper.save_model_and_opt(epoch, ckpt_path, best_model=False,
                                      model=(model.module.state_dict() if cfg.distributed else
                                             model.state_dict()),
                                      optimizer=optimizer.state_dict())
            helper.save_stats(ckpt_path)

        print() if master else None


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    set_seed(cfg.seed)
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)

    # Set-up device
    device, device_str, pin_memory, master, device_id = get_device(distributed=cfg.distributed,
                                                                   init_method=cfg.init_method,
                                                                   dist_backend=cfg.dist_backend)
    print("Device :", device)

    # Initialize model, optimizer and epoch
    model = ParetoStatePredictorMIS(encoder_type=cfg.encoder_type,
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
    ckpt_path = helper.get_checkpoint_path()
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
    model = DDP(model, device_ids=[device_id]) if cfg.distributed else model

    # Initialize dataloaders
    print("N worker dataloader: ", cfg.n_worker_dataloader)
    train_dataset = helper.get_dataset("train",
                                       cfg.dataset.train.from_pid,
                                       cfg.dataset.train.to_pid)
    train_sampler, train_dataloader = None, None

    val_dataset = helper.get_val_dataset(cfg, train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if cfg.distributed and not cfg.validate_on_master \
        else None
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                sampler=val_sampler,
                                shuffle=False,
                                num_workers=cfg.n_worker_dataloader,
                                pin_memory=pin_memory)
    print_validation_machine(master, val_sampler, device_str, cfg.distributed)

    training_loop(cfg, master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader,
                  val_dataset, val_dataloader, model, optimizer, helper, device, pin_memory, ckpt_path)


if __name__ == "__main__":
    main()
