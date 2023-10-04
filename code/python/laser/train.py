import random

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
from laser.utils import update_scores

statscores = StatScores(task='binary', multidim_average='samplewise')

dataset_dict = {}


def set_seed(seed):
    random.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def init_optimizer(cfg, model):
    opt_cls = getattr(optim, cfg.opt.name)
    opt = opt_cls(model.parameters(), lr=cfg.opt.lr)

    return opt


def init_loss_fn(cfg):
    return nn.BCELoss()


def train_step(model, loss_fn, opt, dataloader, num_objs, num_vars, layer_norm_const=100,
               flag_layer_penalty=False, flag_label_penalty=False):
    model.train()
    scores = None

    # Train over this instance
    for bid, batch in enumerate(dataloader):
        nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
                                                      batch['wtlayer'], batch['wtlabel'], batch['label'])

        # Get layer ids of the nodes in the current batch and predict
        lidxs_t = nf[:, 1] * layer_norm_const
        lidxs = list(map(int, lidxs_t.numpy()))
        context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars)
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
        layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
        scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

    return scores.numpy()


def train_loop(cfg, epoch, model, loss_fn, optimizer):
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))

    pids = list(range(cfg.train.from_pid, cfg.train.to_pid))
    random.shuffle(pids)
    idx = 0
    while idx < len(pids):
        to_pid = int(np.max(idx + cfg.train.inst_per_step, len(pids)))
        _pids = pids[idx: to_pid]

        datasets = []
        for pid in _pids:
            if pid not in dataset_dict:
                dataset_dict[pid] = get_dataset(cfg.prob.name, "train", pid)
            datasets.append(dataset_dict[pid])
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

        scores = train_step(model,
                            loss_fn,
                            optimizer,
                            dataloader,
                            cfg.prob.num_objs,
                            cfg.prob.num_vars,
                            layer_norm_const=cfg.prob.layer_norm_const,
                            flag_layer_penalty=cfg.train.flag_layer_penalty,
                            flag_label_penalty=cfg.train.flag_label_penalty)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

        # Print after each batch of instances is processed
        if cfg.train.verbose:
            acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
            print_result(epoch,
                         "Train",
                         pid=pid,
                         acc=acc,
                         correct=correct,
                         total=total,
                         inst_per_step=cfg.train.inst_per_step)

        idx += cfg.train.inst_per_step

    return scores_df, tp, fp, tn, fn


def val_step(model, dataloader, num_objs, num_vars, layer_norm_const=100):
    model.eval()
    scores = None

    with torch.no_grad():
        # Train over this instance
        for bid, batch in enumerate(dataloader):
            nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
                                                          batch['wtlayer'], batch['wtlabel'], batch['label'])

            # Get layer ids of the nodes in the current batch
            lidxs_t = nf[:, 1] * layer_norm_const
            lidxs = list(map(int, lidxs_t.numpy()))
            context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars)

            preds = model(inst_feat, context_feat, nf, pf)

            score = statscores(preds.clone().detach(), label)
            layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
            scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

    return scores.numpy()


def val_loop(cfg, epoch, model):
    rng = random.Random(100)
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])

    pids = list(range(cfg.train.from_pid, cfg.train.to_pid))
    random.shuffle(pids)
    idx = 0
    while idx < len(pids):
        to_pid = int(np.max(idx + cfg.train.inst_per_step, len(pids)))
        _pids = pids[idx: to_pid]

        datasets = []
        for pid in _pids:
            if pid not in dataset_dict:
                dataset_dict[pid] = get_dataset(cfg.prob.name, "val", pid)
            datasets.append(dataset_dict[pid])
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=cfg.val.batch_size, shuffle=True)

        scores = val_step(model,
                          dataloader,
                          cfg.prob.num_objs,
                          cfg.prob.num_vars,
                          layer_norm_const=cfg.prob.layer_norm_const)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)
        idx += cfg.train.inst_per_step

    return scores_df, tp, fp, tn, fn


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):
    # Get model, optimizer and loss function
    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    optimizer = init_optimizer(cfg, model)
    loss_fn = init_loss_fn(cfg)

    best_acc = 0
    for epoch in range(cfg.train.epochs):
        train_result = train_loop(cfg,
                                  epoch,
                                  model,
                                  loss_fn,
                                  optimizer)
        scores_df, tp, fp, tn, fn = train_result

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
            val_result = val_loop(cfg, epoch, model)
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

        print()


if __name__ == "__main__":
    main()
