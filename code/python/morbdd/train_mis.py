import time

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import Meter
from morbdd.utils import setup_device
from morbdd.utils import set_seed
from morbdd.utils.mis import MISTrainingHelper
from morbdd.utils.mis import get_size
from morbdd.utils import dict2cpu
import omegaconf
from torch.utils.data import Subset

import random


def log_dataset(dataset):
    pass


def reduce_stats(data_time, batch_time, losses):
    dist.all_reduce(losses.sum, dist.ReduceOp.SUM)
    dist.all_reduce(data_time.avg, dist.ReduceOp.AVG)
    dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG)
    dist.all_reduce(batch_time.sum, dist.ReduceOp.AVG)


def tensor2np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


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


def process_batch(batch, model, version=1):
    if version == 1:
        objs, adjs, pos, _, lids, vids, states, labels = batch

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        return lids, labels, logits, loss


def get_dataloader(cfg, helper, pin_memory):
    # Initialize dataloaders
    # Train
    print("Num worker dataloader: ", cfg.n_worker)
    train_dataset = helper.get_dataset("train", cfg.dataset.train.from_pid, cfg.dataset.train.to_pid)
    sampler = DistributedSampler(train_dataset, shuffle=True) if cfg.distributed else None
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None),
                                  num_workers=cfg.n_worker, pin_memory=pin_memory)

    # Val
    val_dataset = helper.get_dataset("val", cfg.dataset.val.from_pid, cfg.dataset.val.to_pid)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) \
        if cfg.distributed and not cfg.validate_on_master else None
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=False,
                                num_workers=cfg.n_worker, pin_memory=pin_memory)

    return train_dataloader, val_dataloader


def reset_stats():
    return Meter('DataTime'), Meter('BatchTime'), Meter('Loss'), torch.empty((4, 0))


@torch.no_grad()
def validate(dataloader, model, device, helper, pin_memory=True, master=False, distributed=False,
             validate_on_master=True, best_f1=0):
    model.eval()
    epoch_stats = None
    if (not distributed) or (distributed and validate_on_master and master) or (distributed and not validate_on_master):
        data_time, batch_time, losses, result = (Meter('DataTime'), Meter('BatchTime'), Meter('Loss'),
                                                 torch.empty((4, 0)))

        start_time = time.time()
        for batch in dataloader:
            batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            lids, labels, logits, loss = process_batch(batch, model, version=1)
            
            # Compute stats
            scores = F.softmax(logits, dim=-1)
            preds = torch.argmax(scores, dim=-1)
            batch_result = torch.stack((labels, lids, scores, preds))
            result = torch.stack((result, batch_result), dim=1)

            losses.update(loss.detach(), labels.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
            start_time = time.time()

        if distributed and not validate_on_master:
            reduce_stats(data_time, batch_time, losses, epoch_time)
            result_lst = [result for _ in range(world_size)] if master else None
            dist.gather(result, gather_list=result_lst, dst=0)

        if helper.val_stats[-1]["f1"] > best_f1:
            best_f1 = helper.val_stats[-1]["f1"]
            # Save model
            # Save stats
            print("* Best F1: {}".format(best_f1))

    return epoch_stats


def train(mode, dv, dataloader, model, optimizer, device, helper, iter_num, clip_grad=1.0, norm_type=2.0,
          pin_memory=True, master=True, distributed=False, val_loader=None, validate_on_master=True,
          validate_every=100, best_f1=0):
    model.train()
    data_time, batch_time, losses, result = reset_stats()

    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

        lids, labels, logits, loss = process_batch(batch, model, version=dv)

        # Learn
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=norm_type) if clip_grad > 0 else None
        optimizer.step()

        # Compute stats
        scores = F.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(scores, dim=-1)
        batch_result = torch.stack((labels, lids, scores, preds))
        result = torch.cat((result, batch_result), dim=1)

        losses.update(loss.detach(), labels.shape[0])
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

        iter_num += 1
        if mode == "iter":
            # Validate
            if iter_num % validate_every == 0:
                best_f1 = validate(val_loader, model, device, helper, pin_memory=pin_memory, master=master,
                                   distributed=distributed, validate_on_master=validate_on_master, best_f1=best_f1)

    result = result.T

    return data_time, batch_time, losses, result


def training_loop(cfg, world_size, master, start_epoch, end_epoch, train_loader, val_loader, model, optimizer,
                  helper, device, pin_memory, ckpt_path, best_f1):
    iter_num = 0
    for epoch in range(start_epoch, end_epoch):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if master:
            print("Train samples: {}, Val samples {}".format(len(train_loader.dataset), len(val_loader.dataset)))
            print("Train loader: {}, Val loader {}".format(len(train_loader), len(val_loader)))

        # Train
        data_time, batch_time, loss, result = train(cfg.train_mode, cfg.dataset.version, train_loader, model, optimizer,
                                                    device, helper, iter_num, clip_grad=cfg.clip_grad,
                                                    norm_type=cfg.norm_type,
                                                    pin_memory=pin_memory, distributed=cfg.distributed)
        if cfg.distributed:
            reduce_stats(data_time, batch_time, loss)
            result_lst = [result for _ in range(world_size)] if master else None
            dist.gather(result, gather_list=result_lst, dst=0)

        if master:
            data_time, batch_time, epoch_time, loss, result = map(tensor2np,
                                                                  (data_time.avg, batch_time.avg, batch_time.sum,
                                                                   loss, result))
            # stats = get_stats(result)
            # helper.compute_meta_stats_and_print("train", stats)

        # Validate
        if cfg.train_mode == "epoch":
            if (epoch + 1) % cfg.validate_every_epoch == 0:
                best_f1 = validate(val_loader, model, device, helper, pin_memory=True, master=False, distributed=False,
                                   validate_on_master=True, best_f1=best_f1)

            if master and (epoch + 1) % cfg.save_every == 0:
                helper.save_model_and_opt(epoch, ckpt_path, best_model=False,
                                          model=(model.module.state_dict() if cfg.distributed else
                                                 model.state_dict()),
                                          optimizer=optimizer.state_dict())
                helper.save_stats(ckpt_path)

            print() if master else None

        if cfg.train_mode == "iter" and iter_num > cfg.max_iter:
            break


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    set_seed(cfg.seed)
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)

    # Set/Get checkpoint
    ckpt_path = helper.get_checkpoint_path()
    ckpt_path.mkdir(exist_ok=True, parents=True)
    print("Checkpoint path: {}".format(ckpt_path))
    ckpt = None
    if cfg.init_from == "resume":
        mdl_path = ckpt_path / "model.pt"
        if mdl_path.exists():
            ckpt = torch.load(mdl_path, map_location="cpu")

    # Set device
    world_size, rank, master, device_id, device_str, pin_memory = setup_device(distributed=cfg.distributed,
                                                                               init_method=cfg.init_method,
                                                                               dist_backend=cfg.dist_backend)
    device = torch.device(device_str)
    print("Device: ", device_str)

    # Set model
    model = ParetoStatePredictorMIS(encoder_type=cfg.encoder_type,
                                    n_node_feat=cfg.n_node_feat,
                                    n_edge_type=cfg.n_edge_type,
                                    d_emb=cfg.d_emb,
                                    top_k=cfg.top_k,
                                    n_blocks=cfg.n_blocks,
                                    n_heads=cfg.n_heads,
                                    dropout_token=cfg.dropout_token,
                                    dropout=cfg.dropout,
                                    bias_mha=cfg.bias_mha,
                                    bias_mlp=cfg.bias_mlp,
                                    h2i_ratio=cfg.h2i_ratio)
    model.to(device)
    model = DDP(model, device_ids=[device_id]) if cfg.distributed else model

    # Set optimizer
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)

    # Set dataloader, epoch and metric
    train_loader, val_loader = get_dataloader(cfg, helper, pin_memory)
    start_epoch, end_epoch, best_f1 = 0, cfg.epochs, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["opt_dict"])
        stats = torch.load(ckpt_path / "stats.pt", map_location="cpu")
        for v in stats["val"]:
            if v["f1"] < best_f1:
                best_f1 = v["f1"]
        start_epoch = int(ckpt["epoch"])
    if cfg.train_mode == "iter":
        end_epoch = int(len(train_loader) / cfg.max_iter) + 1

    print_validation_machine(master, val_loader.sampler, device_str, cfg.distributed)

    # Setup wandb
    if cfg.wandb.log and master:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(project=cfg.wandb.project, name=str(ckpt_path))
        log_dataset(train_loader.dataset)
        log_dataset(val_loader.dataset)
        if cfg.wandb.track_grad:
            wandb.watch(model, log_freq=100)

    training_loop(cfg, world_size, master, start_epoch, end_epoch, train_loader, val_loader, model, optimizer,
                  helper, device, pin_memory, ckpt_path, best_f1)


if __name__ == "__main__":
    main()
