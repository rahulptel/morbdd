import random
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import StatScores

from laser.model import model_factory
from laser.utils import calculate_accuracy
from laser.utils import checkpoint
from laser.utils import get_context_features
from laser.utils import get_dataset
from laser.utils import print_result
from laser.utils import set_device
from laser.utils import update_scores
from laser.utils import get_split_datasets

statscores = StatScores(task='binary', multidim_average='samplewise')

dataset_dict = {}
val_dataset = None
val_dataloader = None


def init_optimizer(cfg, model):
    opt_cls = getattr(optim, cfg.opt.name)
    opt = opt_cls(model.parameters(), lr=cfg.opt.lr)

    return opt


def init_loss_fn(cfg):
    return nn.BCELoss()


def train_step(model, loss_fn, opt, dataloader, num_objs, num_vars, device, layer_norm_const=100,
               flag_layer_penalty=False, flag_label_penalty=False):
    model.train()
    scores = None

    num_batches = len(dataloader)
    # Train over this instance
    for bid, batch in enumerate(dataloader):
        sys.stdout.write("\r")
        sys.stdout.write(f"\t\tProgress: {((bid + 1) / num_batches) * 100:.2f}%")
        sys.stdout.flush()

        nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
                                                      batch['wtlayer'], batch['wtlabel'], batch['label'])

        # Get layer ids of the nodes in the current batch and predict
        lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
        lidxs = list(map(int, lidxs_t.cpu().numpy()))
        context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)
        preds = model(inst_feat, context_feat, nf, pf)

        # Weighted loss
        weight = None
        if flag_layer_penalty:
            weight = wtlayer
        if flag_label_penalty:
            weight = wtlabel if weight is None else weight * wtlabel
        loss_fn = nn.BCELoss(weight=weight.reshape(-1, 1))
        loss_batch = loss_fn(preds, label)

        opt.zero_grad()
        loss_batch.backward()
        opt.step()

        score = statscores(preds.clone().detach(), label)
        score = score if len(score.shape) > 1 else score.unsqueeze(0)
        layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
        scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)
    print()

    return scores.cpu().numpy()


def train_loop(cfg, epoch, model, loss_fn, optimizer, device):
    print("\tTrain loop...")
    global dataset_dict
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))

    pids = list(range(cfg.train.from_pid, cfg.train.to_pid))
    num_pids = len(pids)
    random.shuffle(pids)
    idx = 0
    while idx < num_pids:
        to_pid = int(np.min([idx + cfg.train.inst_per_step, num_pids]))
        _pids = pids[idx: to_pid]

        datasets, dataset_dict = get_split_datasets(_pids, cfg.prob.name, cfg.prob.size, "train",
                                                    cfg.train.neg_pos_ratio,
                                                    cfg.train.min_samples,
                                                    dataset_dict,
                                                    device=device)
        if len(datasets) == 0:
            continue
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
        print(f"\t\tDataset size: {len(dataset)}")
        print(f"\t\tNumber of batches: {len(dataloader)}")
        scores = train_step(model,
                            loss_fn,
                            optimizer,
                            dataloader,
                            cfg.prob.num_objs,
                            cfg.prob.num_vars,
                            device,
                            layer_norm_const=cfg.prob.layer_norm_const,
                            flag_layer_penalty=cfg.train.flag_layer_penalty,
                            flag_label_penalty=cfg.train.flag_label_penalty)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

        # Print after each batch of instances is processed
        if cfg.train.verbose:
            acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
            print_result(epoch,
                         "Train",
                         pid=idx,
                         acc=acc,
                         correct=correct,
                         total=total,
                         inst_per_step=cfg.train.inst_per_step)

        idx += cfg.train.inst_per_step

    return scores_df, tp, fp, tn, fn


def val_step(model, dataloader, num_objs, num_vars, device, layer_norm_const=100):
    model.eval()
    scores = None
    num_batches = len(dataloader)
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            sys.stdout.write("\r")
            sys.stdout.write(f"\t\tProgress: {((bid + 1) / num_batches) * 100:.2f}%")
            sys.stdout.flush()

            nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
                                                          batch['wtlayer'], batch['wtlabel'], batch['label'])

            # Get layer ids of the nodes in the current batch
            lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
            lidxs = list(map(int, lidxs_t.cpu().numpy()))
            context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)

            preds = model(inst_feat, context_feat, nf, pf)

            score = statscores(preds.clone().detach(), label)
            score = score if len(score.shape) > 1 else score.unsqueeze(0)
            layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
            scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)
    print()

    return scores.cpu().numpy()


def val_loop(cfg, epoch, model, device):
    global dataset_dict
    print("\tValidation loop...")
    global val_dataloader, val_dataset

    rng = random.Random(100)
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])

    pids = list(range(cfg.val.from_pid, cfg.val.to_pid))
    if val_dataloader is None:
        datasets, dataset_dict = get_split_datasets(pids, cfg.prob.name, cfg.prob.size, "val",
                                                    cfg.val.neg_pos_ratio,
                                                    cfg.val.min_samples,
                                                    dataset_dict,
                                                    device)
        val_dataset = ConcatDataset(datasets)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val.batch_size, shuffle=False)

    print(f"\t\tDataset size: {len(val_dataset)}")
    print(f"\t\tNumber of batches: {len(val_dataloader)}")
    scores = val_step(model,
                      val_dataloader,
                      cfg.prob.num_objs,
                      cfg.prob.num_vars,
                      device,
                      layer_norm_const=cfg.prob.layer_norm_const)
    scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

    return scores_df, tp, fp, tn, fn


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):
    # Set device
    device = set_device(cfg.device)

    # Get model, optimizer and loss function
    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    model.to(device)

    optimizer = init_optimizer(cfg, model)
    loss_fn = init_loss_fn(cfg)

    best_acc = 0
    for epoch in range(cfg.train.epochs):
        ep_start = time.time()
        print(f"Epoch {epoch}")
        train_result = train_loop(cfg,
                                  epoch,
                                  model,
                                  loss_fn,
                                  optimizer,
                                  device)
        scores_df, tp, fp, tn, fn = train_result
        train_end = time.time()
        print("\t\tTraining time ", train_end - ep_start)

        # Log training accuracy metrics
        acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
        print_result(epoch,
                     "Train",
                     acc=acc,
                     correct=correct,
                     total=total)
        checkpoint(cfg,
                   "train",
                   epoch=epoch,
                   model=model,
                   scores_df=scores_df)

        if (epoch + 1) % cfg.val.every == 0:
            val_start = time.time()
            val_result = val_loop(cfg, epoch, model, device)
            val_end = time.time()
            print("\t\tValidation time ", val_end - val_start)

            scores_df, tp, fp, tn, fn = val_result
            acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc

            # Log training accuracy metrics
            print_result(epoch,
                         "Val",
                         acc=acc,
                         correct=correct,
                         total=total,
                         is_best=is_best)

            checkpoint(cfg,
                       "val",
                       epoch=epoch,
                       model=model,
                       scores_df=scores_df,
                       is_best=is_best)

        ep_end = time.time()

        print("\tEpoch time: ", ep_end - ep_start)


if __name__ == "__main__":
    main()
