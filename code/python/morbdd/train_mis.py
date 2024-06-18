import random
import time

import hydra
import omegaconf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import Meter
from morbdd.utils import all_reduce
from morbdd.utils import set_seed
from morbdd.utils import setup_device
from morbdd.utils import tensor2np
from morbdd.utils.mis import MISTrainingHelper
from morbdd.utils.mis import get_size


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
    return Meter('DataTime'), Meter('BatchTime'), Meter('Loss'), torch.empty((6, 0))


def process_batch(batch, model, loss_fn, version=1):
    objs, adjs, pos, _, lids, vids, states, labels = batch
    if version == 1:
        # Get logits and compute loss
        logits = model(objs, adjs, pos, lids, vids, states)
        logits, labels = logits.reshape(-1, 2), labels.long().reshape(-1)
        loss = loss_fn(logits, labels)

        # Get predictions
        scores = F.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(scores, dim=-1)
        result = torch.stack((labels, lids, logits[:, 0], logits[:, 1], scores[:, 1], preds))

        return loss, result


def aggregate_distributed_stats(losses, data_time, batch_time):
    all_reduce(losses.sum, "sum")
    all_reduce(losses.count, "sum")
    losses.avg = losses.sum / losses.count

    all_reduce(data_time.avg, "avg")
    all_reduce(batch_time.avg, "avg")
    all_reduce(batch_time.sum, "avg")


def get_result_stats(result):
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(result[:, 0], result[:, -2])
    return {
        "precision": precision_score(result[:, 0], result[:, -1]),
        "recall": recall_score(result[:, 0], result[:, -1]),
        "f1": f1_score(result[:, 0], result[:, -1]),
        "roc_score": roc_auc_score(result[:, 0], result[:, -2]),
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_thresholds": roc_thresholds
    }


def post_process_distributed(losses, data_time, batch_time, result, world_size):
    aggregate_distributed_stats(losses, data_time, batch_time)
    result_lst = [result for _ in range(world_size)]
    dist.gather(result, gather_list=result_lst, dst=0)
    result = torch.cat(result_lst)

    return result


def set_out(master, losses, data_time, batch_time, result, out, helper, split):
    if master:
        # Convert to numpy
        r = list(map(tensor2np, (losses.avg, data_time.avg, batch_time.avg, batch_time.sum, result)))
        out.update(get_result_stats(result))
        out["loss"], out["data_time"], out["batch_time"], out["epoch_time"] = r[0], r[1], r[2], r[3]
        getattr(helper, f"{split}_stats").append(out)


def save(helper, ckpt_path, epoch, iter_num, model, optimizer, is_best):
    raw_model = model.module if hasattr(model, "module") else model
    helper.save_model_and_opt(epoch, iter_num, ckpt_path, best_model=is_best, model=raw_model.state_dict(),
                              optimizer=optimizer.state_dict())
    helper.save_stats(ckpt_path)


# def log_to_wandb(mode, split, helper):
#     if split == "train" and len(helper.train_stats):
#         wandb.log(helper.train_stats)
#
#     if split == "val" and len(helper.val_stats):
#         pass


@torch.no_grad()
def validate(dataloader, model, loss_fn, device, helper, pin_memory=True, master=False, distributed=False, world_size=1,
             validate_on_master=True, wandb_log=False):
    if (not distributed) or (distributed and validate_on_master and master) or (distributed and not validate_on_master):
        model.eval()

        out = {}
        data_time, batch_time, losses, result = reset_stats()

        start_time = time.time()
        for batch in dataloader:
            batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            loss, batch_result = process_batch(batch, model, loss_fn, version=1)
            result = torch.cat((result, batch_result), dim=1)

            losses.update(loss.detach(), batch_result.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
            start_time = time.time()

        result = result.T
        if distributed and not validate_on_master:
            result = post_process_distributed(losses, data_time, batch_time, result, world_size)
        set_out(master, losses, data_time, batch_time, result, out, helper, "val")

        model.train()

        if wandb_log:
            wandb.log({
                "val/loss": helper.val_stats[-1]["loss"],
                "val/f1": helper.val_stats[-1]["f1"],
                "val/recall": helper.val_stats[-1]["recall"],
                "val/precision": helper.val_stats[-1]["precision"],
            }, commit=False)


# def train(mode="epoch", dv=1, train_loader=None, model=None, optimizer=None, loss_fn="bce", device="cpu", helper=None,
#           clip_grad=1.0, norm_type=2.0, pin_memory=True, world_size=1, master=True, distributed=False, epoch=0,
#           iter_num=0, max_iter=100, val_loader=None, validate_on_master=True, validate_every=100, best_f1=0,
#           ckpt_path=None, wandb_log=False):
#     model.train()
#
#     out = {}
#     data_time, batch_time, losses, result = reset_stats()
#
#     start_time = time.time()
#     for batch_id, batch in enumerate(train_loader):
#         batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
#         data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
#
#         loss, batch_result = process_batch(batch, model, loss_fn, version=dv)
#         # Learn
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=norm_type) if clip_grad > 0 else None
#         optimizer.step()
#
#         result = torch.cat((result, batch_result), dim=1)
#         losses.update(loss.detach(), batch_result.shape[0])
#         batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
#         start_time = time.time()
#
#         iter_num += 1
#         if mode == "iter" and iter_num % validate_every == 0:
#             validate(val_loader, model, loss_fn, device, helper, pin_memory=pin_memory, master=master,
#                      distributed=distributed, world_size=world_size, validate_on_master=validate_on_master)
#             if master and helper.val_stats[-1]["f1"] > best_f1:
#                 best_f1 = helper.val_stats[-1]["f1"]
#                 save(helper, ckpt_path, epoch, iter_num, model, optimizer, True)
#                 print("* Best F1: {}".format(best_f1))
#
#         if mode == "iter" and iter_num > max_iter:
#             break
#
#     result = result.T
#     if distributed:
#         result = post_process_distributed(losses, data_time, batch_time, result, world_size)
#     set_out(master, losses, data_time, batch_time, result, out, helper, "train")
#
#     return iter_num, best_f1


# def training_loop(cfg, world_size, master, start_epoch, end_epoch, iter_num, train_loader, val_loader, model, optimizer,
#                   loss_fn, helper, device, pin_memory, ckpt_path, best_f1):
#     if master:
#         print("Train samples: {}, Val samples {}".format(len(train_loader.dataset), len(val_loader.dataset)))
#         print("Train loader: {}, Val loader {}".format(len(train_loader), len(val_loader)))
#
#     for epoch in range(start_epoch, end_epoch):
#         if cfg.distributed:
#             train_loader.sampler.set_epoch(epoch)
#
#         if cfg.train_mode == "epoch" and epoch % cfg.validate_every_epoch == 0:
#             validate(val_loader, model, loss_fn, device, helper, pin_memory=pin_memory, master=master,
#                      distributed=cfg.distributed, world_size=world_size, validate_on_master=cfg.validate_on_master)
#             if master:
#                 if cfg.wandb_log:
#                     log_to_wandb(helper)
#
#                 if helper.val_stats[-1]["f1"] > best_f1:
#                     best_f1 = helper.val_stats[-1]["f1"]
#                     save(helper, ckpt_path, epoch, iter_num, model, optimizer, True)
#                     print("* Best F1: {}".format(best_f1))
#
#         if master and epoch % cfg.save_every == 0:
#             save(helper, ckpt_path, epoch, iter_num, model, optimizer, False)
#
#         if cfg.train_mode == "iter" and iter_num > cfg.max_iter:
#             break
#
#         # Train
#         out = train(mode=cfg.train_mode, dv=cfg.dataset.version, train_loader=train_loader, model=model,
#                     optimizer=optimizer, loss_fn=loss_fn, device=device, helper=helper, clip_grad=cfg.clip_grad,
#                     norm_type=cfg.norm_type, pin_memory=pin_memory, world_size=world_size, master=master,
#                     distributed=cfg.distributed, iter_num=iter_num, max_iter=cfg.max_iter, val_loader=val_loader,
#                     validate_on_master=cfg.validate_on_master, validate_every=cfg.validate_every_iter,
#                     ckpt_path=ckpt_path, best_f1=best_f1, wandb_log=cfg.wandb.log)
#         iter_num, best_f1 = out
#
#         print() if master else None


def iter_loop(cfg, world_size, master, iter_num, train_loader, val_loader, model, optimizer, loss_fn, helper, device,
              pin_memory, ckpt_path, best_f1):
    if master:
        print("Train samples: {}, Val samples {}".format(len(train_loader.dataset), len(val_loader.dataset)))
        print("Train loader: {}, Val loader {}".format(len(train_loader), len(val_loader)))

    if iter_num == 0:
        validate(val_loader, model, loss_fn, device, helper, pin_memory=pin_memory, master=master,
                 distributed=cfg.distributed, world_size=world_size, validate_on_master=cfg.validate_on_master)
        if master and helper.val_stats[-1]["f1"] > best_f1:
            best_f1 = helper.val_stats[-1]["f1"]
            save(helper, ckpt_path, -1, iter_num, model, optimizer, True)
            print("* Best F1: {}".format(best_f1))

    while True:
        # Shuffle train loader
        train_loader.sampler.set_epoch(random.randint(0, 100000))
        out = {}
        data_time, batch_time, losses, result = reset_stats()

        start_time = time.time()
        for batch_id, batch in enumerate(train_loader):
            batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            loss, batch_result = process_batch(batch, model, loss_fn, version=cfg.dataset.version)
            # Learn
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad,
                                           norm_type=cfg.norm_type) if cfg.clip_grad > 0 else None
            optimizer.step()

            result = torch.cat((result, batch_result), dim=1)
            losses.update(loss.detach(), batch_result.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            iter_num += 1
            if iter_num % cfg.validate_every == 0:
                # Get val stats
                validate(val_loader, model, loss_fn, device, helper, pin_memory=pin_memory, master=master,
                         distributed=cfg.distributed, world_size=world_size, validate_on_master=cfg.validate_on_master)
                if master and helper.val_stats[-1]["f1"] > best_f1:
                    best_f1 = helper.val_stats[-1]["f1"]
                    # save(helper, ckpt_path, -1, iter_num, model, optimizer, True)
                    raw_model = model.module if hasattr(model, "module") else model
                    helper.save_model_and_opt(-1, iter_num, ckpt_path, best_model=True, model=raw_model.state_dict(),
                                              optimizer=optimizer.state_dict())
                    print("* Best F1: {}".format(best_f1))

                # Get train stats
                result = result.T
                result = post_process_distributed(losses, data_time, batch_time, result, world_size) \
                    if cfg.distributed else result
                set_out(master, losses, data_time, batch_time, result, out, helper, "train")
                save(helper, ckpt_path, -1, iter_num, model, optimizer, False)
                if cfg.wandb.log:
                    wandb.log({
                        "train/loss": helper.train_stats[-1]["loss"],
                        "train/f1": helper.train_stats[-1]["f1"],
                        "train/recall": helper.train_stats[-1]["recall"],
                        "train/precision": helper.train_stats[-1]["precision"],
                    })

            print() if master else None
            if iter_num > cfg.max_iter:
                return

            start_time = time.time()


def epoch_loop(cfg, world_size, master, start_epoch, end_epoch, iter_num, train_loader, val_loader, model, optimizer,
               loss_fn, helper, device, pin_memory, ckpt_path, best_f1):
    def train(dv=1, clip_grad=1.0, norm_type=2.0, distributed=False, wandb_log=False):
        model.train()

        out = {}
        data_time, batch_time, losses, result = reset_stats()

        start_time = time.time()
        for batch_id, batch in enumerate(train_loader):
            batch = [item.to(device, non_blocking=True) for item in batch] if pin_memory else batch
            data_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))

            loss, batch_result = process_batch(batch, model, loss_fn, version=dv)
            # Learn
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad,
                                           norm_type=norm_type) if clip_grad > 0 else None
            optimizer.step()

            result = torch.cat((result, batch_result), dim=1)
            losses.update(loss.detach(), batch_result.shape[0])
            batch_time.update(torch.tensor(time.time() - start_time, dtype=torch.float32, device=device))
            start_time = time.time()

        result = result.T
        result = post_process_distributed(losses, data_time, batch_time, result, world_size) if distributed else result
        set_out(master, losses, data_time, batch_time, result, out, helper, "train")

        if wandb_log:
            wandb.log({
                "train/loss": helper.train_stats[-1]["loss"],
                "train/f1": helper.train_stats[-1]["f1"],
                "train/recall": helper.train_stats[-1]["recall"],
                "train/precision": helper.train_stats[-1]["precision"],
            })

    if master:
        print("Train samples: {}, Val samples {}".format(len(train_loader.dataset), len(val_loader.dataset)))
        print("Train loader: {}, Val loader {}".format(len(train_loader), len(val_loader)))

    for epoch in range(start_epoch, end_epoch):
        train_loader.sampler.set_epoch(epoch) if cfg.distributed else None

        if epoch % cfg.validate_every_epoch == 0:
            validate(val_loader, model, loss_fn, device, helper, pin_memory=pin_memory, master=master,
                     distributed=cfg.distributed, world_size=world_size, validate_on_master=cfg.validate_on_master)
            if master:
                if cfg.wandb_log:
                    # log_to_wandb(helper)
                    pass

                if helper.val_stats[-1]["f1"] > best_f1:
                    best_f1 = helper.val_stats[-1]["f1"]
                    # save(helper, ckpt_path, epoch, iter_num, model, optimizer, True)
                    raw_model = model.module if hasattr(model, "module") else model
                    helper.save_model_and_opt(-1, iter_num, ckpt_path, best_model=True, model=raw_model.state_dict(),
                                              optimizer=optimizer.state_dict())
                    print("* Best F1: {}".format(best_f1))

        if master and epoch % cfg.save_every == 0:
            save(helper, ckpt_path, epoch, iter_num, model, optimizer, False)

        # Train
        train(dv=cfg.dataset.version, clip_grad=cfg.clip_grad, norm_type=cfg.norm_type, distributed=cfg.distributed,
              wandb_log=cfg.wandb.log)

        print() if master else None


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

    # Set optimizer and loss function
    opt_cls = getattr(optim, cfg.opt.name)
    optimizer = opt_cls(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.wd)
    loss_fn = None
    if cfg.loss == "bce":
        loss_fn = F.cross_entropy
    assert loss_fn is not None

    # Set dataloader, epoch and metric
    train_loader, val_loader = get_dataloader(cfg, helper, pin_memory)
    start_epoch, end_epoch, iter_num, best_f1 = 0, cfg.epochs, 0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["opt_dict"])
        start_epoch = int(ckpt["epoch"])
        iter_num = int(ckpt["iter"])

        stats = torch.load(ckpt_path / "stats.pt", map_location="cpu")
        for v in stats["val"]:
            if v["f1"] < best_f1:
                best_f1 = v["f1"]

    print_validation_machine(master, val_loader.sampler, device_str, cfg.distributed)

    # Setup wandb
    if cfg.wandb.log and master:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(project=cfg.wandb.project, name=str(ckpt_path))

        if cfg.wandb.track_grad > 0:
            wandb.watch(model, log_freq=cfg.wandb.track_grad)

    if cfg.train_mode == "epoch":
        epoch_loop(cfg, world_size, master, start_epoch, end_epoch, iter_num, train_loader, val_loader, model,
                   optimizer, loss_fn, helper, device, pin_memory, ckpt_path, best_f1)
    elif cfg.train_mode == "iter":
        iter_loop(cfg, world_size, master, iter_num, train_loader, val_loader, model, optimizer, loss_fn, helper,
                  device, pin_memory, ckpt_path, best_f1)


if __name__ == "__main__":
    main()
