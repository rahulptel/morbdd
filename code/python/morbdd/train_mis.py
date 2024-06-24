import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import Meter
from morbdd.utils import get_device
from morbdd.utils import set_seed
from morbdd.utils.mis import MISTrainingHelper
from morbdd.utils.mis import get_size
from torch.utils.tensorboard import SummaryWriter


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


def get_stats(losses, data_time, batch_time, result):
    result = result.cpu().numpy()

    l0c, l0b = np.histogram(result[:, 2], 10)
    l1c, l1b = np.histogram(result[:, 3], 10)
    s0c, s0b = np.histogram(result[result[:, 0] == 0][:, -2], 10)
    s1c, s1b = np.histogram(result[result[:, 0] == 0][:, -2], 10)
    stats = {
        "loss": losses.avg.cpu().numpy(),
        "data_time": data_time.avg.cpu().numpy(),
        "batch_time": batch_time.avg.cpu().numpy(),
        "epoch_time": batch_time.sum.cpu().numpy(),
        "f1": f1_score(result[:, 0], result[:, -1]),
        "recall": recall_score(result[:, 0], result[:, -1]),
        "precision": precision_score(result[:, 0], result[:, -1]),
        "acc": np.sum((result[:, 0] == result[:, -1])) / result.shape[0],
        "specificity": (result[result[:, 0] == 0][:, -1] == 0).sum(),
        "lgt0-mean": np.mean(result[:, 2]),
        "lgt0-std": np.std(result[:, 2]),
        "lgt0-med": np.median(result[:, 2]),
        "lgt0-min": np.min(result[:, 2]),
        "lgt0-max": np.max(result[:, 2]),
        "lgt0-count": l0c,
        "lgt0-bins": l0b,
        "score0-count": s0c,
        "score0-bins": s0b,
        "lgt1-mean": np.mean(result[:, 3]),
        "lgt1-std": np.std(result[:, 3]),
        "lgt1-med": np.median(result[:, 3]),
        "lgt1-min": np.min(result[:, 3]),
        "lgt1-max": np.max(result[:, 3]),
        "lgt1-count": l1c,
        "lgt1-bins": l1b,
        "score1-count": s1c,
        "score1-bins": s1b
    }

    return stats


def aggregate_distributed_stats(losses=None, data_time=None, batch_time=None, result=None, master=True):
    if losses is not None:
        dist.all_reduce(losses.sum, dist.ReduceOp.SUM)
        losses.count = torch.tensor(losses.count).to(losses.sum.device)
        dist.all_reduce(losses.count, dist.ReduceOp.SUM)
        losses.avg = losses.sum / losses.count
    if data_time is not None:
        dist.all_reduce(data_time.avg, dist.ReduceOp.AVG)
    if batch_time is not None:
        dist.all_reduce(batch_time.avg, dist.ReduceOp.AVG)
        dist.all_reduce(batch_time.sum, dist.ReduceOp.AVG)
    if result is not None:
        result_lst = [torch.zeros_like(result) for _ in range(dist.get_world_size())] if master else None
        dist.gather(result, gather_list=result_lst, dst=0)
        result = torch.cat(result_lst) if master else master

    return result


def process_batch(batch, model, loss_fn, version=1, epoch=0, batch_id=0, max_batches=1, writer=None, log=False,
                  split="train"):
    objs, adjs, pos, _, lids, vids, states, labels = batch
    objs = objs / 100
    curr_iter = (epoch * max_batches) + batch_id

    if version == 1:
        # Get logits and compute loss
        # logits = model(objs, adjs, pos, lids, vids, states)
        # ----------------------------------------------------------------------
        n_emb, e_emb = model.token_emb(objs, adjs.int(), pos.float())
        if log and writer is not None:
            writer.add_histogram("token_emb/n/" + split, n_emb, curr_iter)
            writer.add_histogram("token_emb/e/" + split, e_emb, curr_iter)

        n_emb = model.node_encoder(n_emb, e_emb)
        # Instance embedding
        # B x d_emb
        inst_emb = model.graph_encoder(n_emb.sum(1))
        # Layer-index embedding
        # B x d_emb
        li_emb = model.layer_index_encoder(lids.reshape(-1, 1).float())
        # Layer-variable embedding
        # B x d_emb
        # lv_emb = torch.stack([n_emb[pid, vid] for pid, vid in zip(pids_index, vids.int())])
        lv_emb = torch.stack([n_emb[pid, vid] for pid, vid in enumerate(vids.int())])
        # State embedding
        state_emb = torch.stack([n_emb[pid, state].sum(0) for pid, state in enumerate(states.bool())])
        # state_emb = torch.stack([n_emb[pid][state].sum(0) for pid, state in zip(pids_index, indices)])
        # state_emb = torch.stack(state_emb)
        state_emb = model.aggregator(state_emb)
        state_emb = state_emb + inst_emb + li_emb + lv_emb
        # Pareto-state predictor
        logits = model.predictor(model.ln(state_emb))
        # ----------------------------------------------------------------------

        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = loss_fn(logits, labels)

        # Get predictions
        logits = logits.detach()
        scores = F.softmax(logits, dim=-1)
        preds = torch.argmax(scores, dim=-1)
        batch_result = torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))

        return loss, batch_result


def get_grad_norm(model, norm=2):
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(norm=2)

    return total ** (1 / norm)


@torch.no_grad()
def validate(dataloader, model, device, helper, pin_memory=True, master=False, distributed=False,
             validate_on_master=True, epoch=None, writer=None, log_every=1e10):
    stats = {}
    result = None
    if (not distributed) or (distributed and validate_on_master and master) or (distributed and not validate_on_master):
        data_time, batch_time, losses = Meter('DataTime'), Meter('BatchTime'), Meter('Loss')
        result = torch.empty((6, 0)).to(device)
        max_batches = len(dataloader)

        model.eval()
        start_time = time.time()
        for batch_id, batch in enumerate(dataloader):
            if pin_memory:
                batch = [item.to(device, non_blocking=True) for item in batch]
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
            log = master and log_every % batch_id == 0

            loss, batch_result = process_batch(batch, model, F.cross_entropy, epoch=epoch, batch_id=batch_id,
                                               max_batches=max_batches, writer=writer, log=log, split="val")

            result = torch.cat((result, batch_result), dim=1)
            losses.update(loss.detach(), batch_result.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
            start_time = time.time()

        result = result.T
        if distributed and not validate_on_master:
            result = aggregate_distributed_stats(losses=losses, data_time=data_time, batch_time=batch_time,
                                                 result=result, master=master)
        result = result.cpu().numpy()
        stats = get_stats(losses, data_time, batch_time, result)

    return stats, result


def train(dataloader, model, optimizer, device, helper, clip_grad=1.0, norm_type=2.0, pin_memory=True,
          distributed=False, master=True, epoch=None, writer=None, log_every=1e10):
    data_time, batch_time, losses = Meter('DataTime'), Meter('BatchTime'), Meter('Loss')
    result = torch.empty((6, 0)).to(device)
    max_batches = len(dataloader)

    model.train()
    start_time = time.time()
    for batch_id, batch in enumerate(dataloader):
        if pin_memory:
            batch = [item.to(device, non_blocking=True) for item in batch]
        data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        log = master and log_every % batch_id == 0

        # Get logits and compute loss
        loss, batch_result = process_batch(batch, model, F.cross_entropy, epoch=epoch, batch_id=batch_id,
                                           max_batches=max_batches, writer=writer, log=log, split="train")
        # Learn
        optimizer.zero_grad()
        loss.backward()
        if log:
            norm = get_grad_norm(model)
            writer.add_scalar("grad_norm", model.parameters())
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=norm_type)
        optimizer.step()

        result = torch.cat((result, batch_result), dim=1)
        losses.update(loss.detach(), batch_result.shape[0])
        batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
        start_time = time.time()

    result = result.T
    if distributed:
        result = aggregate_distributed_stats(losses=losses, data_time=data_time, batch_time=batch_time, result=result,
                                             master=master)
    result = result.cpu().numpy()
    stats = get_stats(losses, data_time, batch_time, result)

    return stats, result


def training_loop(cfg, master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader, val_dataset,
                  val_sampler, val_dataloader, model, optimizer, helper, device, pin_memory, ckpt_path, world_size,
                  writer, log_every):
    best_f1 = 0
    if master:
        print("Train samples: {}, Val samples {}".format(len(train_dataset), len(val_dataset)))
        print("Train loader: {}, Val loader {}".format(len(train_dataloader), len(val_dataloader)))

    for epoch in range(start_epoch, end_epoch):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        # Train
        start_time = time.time()
        stats, result = train(train_dataloader, model, optimizer, device, helper, clip_grad=cfg.clip_grad,
                              norm_type=cfg.norm_type, pin_memory=pin_memory, distributed=cfg.distributed,
                              master=master, epoch=epoch, writer=writer, log_every=log_every)
        epoch_time = time.time() - start_time
        epoch_time = reduce_epoch_time(epoch_time, device) if cfg.distributed else epoch_time

        if master:
            # stats = dict2cpu(stats) if "cpu" not in str(device) else stats
            epoch_time = float(epoch_time.cpu().numpy()) if cfg.distributed else epoch_time
            stats.update({"epoch_time": epoch_time, "epoch": epoch + 1})
            # stats = helper.compute_meta_stats_and_print("train", stats)
            helper.train_stats.append(stats)
            print("Result shape: ", result.shape)
            helper.print_stats("train", stats)
            print("lgt0: ", stats["lgt0-mean"], stats["lgt0-min"], stats["lgt0-max"])
            print("lgt1: ", stats["lgt1-mean"], stats["lgt1-min"], stats["lgt1-max"])

        # Validate
        if (epoch + 1) % cfg.validate_every == 0:
            stats = {"epoch": epoch + 1}
            for split in cfg.validate_on_split:
                start_time = time.time()
                dataloader = val_dataloader if split == "val" else train_dataloader
                new_stats, result = validate(dataloader, model, device, helper, pin_memory=pin_memory, master=master,
                                             distributed=cfg.distributed, validate_on_master=cfg.validate_on_master,
                                             epoch=epoch, writer=writer, log_every=log_every)
                epoch_time = time.time() - start_time
                epoch_time = reduce_epoch_time(epoch_time, device) if cfg.distributed and not cfg.validate_on_master \
                    else epoch_time

                if master:
                    # new_stats = dict2cpu(new_stats) if "cpu" not in str(device) else new_stats
                    epoch_time = float(epoch_time.cpu().numpy()) if cfg.distributed and not cfg.validate_on_master \
                        else epoch_time
                    new_stats.update({"epoch_time": epoch_time})

                    prefix = "tr_" if split == "train" else ""
                    new_stats = {prefix + k: v for k, v in new_stats.items()} if split == "train" else new_stats
                    stats.update(new_stats)

                    print("Result shape: ", result.shape)
                    helper.print_stats("val", stats, prefix=prefix)
                    print("lgt0: ", stats[prefix + "lgt0-mean"], stats[prefix + "lgt0-min"], stats[prefix + "lgt0-max"])
                    print("lgt1: ", stats[prefix + "lgt1-mean"], stats[prefix + "lgt1-min"], stats[prefix + "lgt1-max"])
                    if split == "val" and stats["f1"] > best_f1:
                        best_f1 = stats["f1"]
                        helper.save_model_and_opt(epoch, ckpt_path, best_model=True,
                                                  model=(model.module.state_dict() if cfg.distributed else
                                                         model.state_dict()),
                                                  optimizer=optimizer.state_dict())
                        helper.save_stats(ckpt_path)
                        print("* Best F1: {}".format(best_f1))

            helper.val_stats.append(stats)

        if master and (epoch + 1) % cfg.save_every == 0:
            helper.save_model_and_opt(epoch, ckpt_path, best_model=False,
                                      model=(model.module.state_dict() if cfg.distributed else
                                             model.state_dict()),
                                      optimizer=optimizer.state_dict())
            helper.save_stats(ckpt_path)

        print() if master else None


def train_test(model, dataloader):
    losses = Meter('Losses')
    result = torch.empty((6, 0))

    model.train()
    for batch_id, batch in enumerate(dataloader):
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        logits = logits.detach()
        scores = F.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(scores, dim=-1)

        losses.update(loss.detach(), labels.shape[0])
        result = torch.cat((result,
                            torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))), dim=1)
        break
    return result.T.cpu().numpy()


@torch.no_grad()
def val_test(model, dataloader):
    losses = Meter('Losses')
    result = torch.empty((6, 0))

    model.eval()
    for batch_id, batch in enumerate(dataloader):
        objs, adjs, pos, _, lids, vids, states, labels = batch
        objs = objs / 100

        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        # Compute stats
        logits = logits.detach()
        scores = F.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(scores, dim=-1)

        losses.update(loss.detach(), labels.shape[0])
        result = torch.cat((result,
                            torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))), dim=1)
        break

    model.train()
    return result.T.cpu().numpy()


def debug(cfg, model, dataset):
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.n_worker_dataloader)

    result_tr = train_test(model, dataloader)
    result_val = val_test(model, dataloader)
    print(result_tr.shape, result_val.shape)
    print(np.array_equal(result_tr, result_val))
    print("Same labels:", np.array_equal(result_tr[:, 0], result_val[:, 0]))
    print("Same lids:", np.array_equal(result_tr[:, 1], result_val[:, 1]))
    print("Same lgt1:", np.array_equal(result_tr[:, 2], result_val[:, 2]))


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    set_seed(cfg.seed)
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)

    # Set-up device
    device, device_str, pin_memory, master, device_id, world_size = get_device(distributed=cfg.distributed,
                                                                               init_method=cfg.init_method,
                                                                               dist_backend=cfg.dist_backend)
    print("Device :", device)

    # Initialize model, optimizer and epoch
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
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)
    start_epoch, end_epoch, best_f1 = 0, cfg.epochs, 0
    # Load model if restarting
    ckpt_path = helper.get_checkpoint_path()
    ckpt_path.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(ckpt_path)
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
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if cfg.distributed else None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=cfg.n_worker_dataloader)

    val_dataset = helper.get_dataset("val",
                                     cfg.dataset.val.from_pid,
                                     cfg.dataset.val.to_pid)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if cfg.distributed and not cfg.validate_on_master \
        else None
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.batch_size,
                                sampler=val_sampler,
                                shuffle=False,
                                num_workers=cfg.n_worker_dataloader,
                                pin_memory=pin_memory)
    print_validation_machine(master, val_sampler, device_str, cfg.distributed)

    # debug(cfg, model, train_dataset)
    training_loop(cfg, master, start_epoch, end_epoch, train_dataset, train_sampler, train_dataloader,
                  val_dataset, val_sampler, val_dataloader, model, optimizer, helper, device, pin_memory, ckpt_path,
                  world_size, writer, cfg.log_every)


if __name__ == "__main__":
    main()
