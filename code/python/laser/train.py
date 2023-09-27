import random

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import StatScores

from laser.data import dataloader_factory
from laser.model import model_factory
from laser.utils import checkpoint, update_scores, calculate_accuracy, print_result

statscores = StatScores(task='binary', multidim_average='samplewise')


def init_optimizer(cfg, model):
    opt_cls = getattr(optim, cfg.opt.name)
    opt = opt_cls(model.parameters(), lr=cfg.opt.lr)

    return opt


def init_loss_fn(cfg):
    return nn.BCELoss()


def get_context_features(layer_idxs, inst_feat, num_objs, num_vars):
    max_lidx = np.max(layer_idxs)
    context = []
    for inst_idx, lidx in enumerate(layer_idxs):
        _inst_feat = inst_feat[inst_idx, :lidx, :]

        ranks = (torch.arange(lidx).reshape(-1, 1) + 1) / num_vars
        _context = torch.concat((_inst_feat, ranks), axis=1)

        ranks_pad = torch.zeros(max_lidx - _inst_feat.shape[0], num_objs + 2)
        _context = torch.concat((_context, ranks_pad), axis=0)

        context.append(_context)
    context = torch.stack(context)

    return context


def train_step(model, loss_fn, opt, dataloader, num_objs, num_vars, layer_norm_const=100):
    model.train()
    scores = None

    # Train over this instance
    for bid, batch in enumerate(dataloader):
        nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
                                                      batch['wtlayer'], batch['wtlabel'], batch['label'])

        # Get layer ids of the nodes in the current batch
        lidxs_t = nf[:, 1] * layer_norm_const
        lidxs = list(map(int, lidxs_t.numpy()))
        context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars)

        preds = model(inst_feat, context_feat, nf, pf)
        loss_batch = loss_fn(preds, label)
        opt.zero_grad()
        loss_batch.backward()
        opt.step()

        score = statscores(preds.clone().detach(), label)
        layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
        scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

    return scores.numpy()


def train_loop(cfg, epoch, model, loss_fn, optimizer, layer_norm_const=100, verbose=True):
    rng = random.Random(100)
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))

    for idx, pid in enumerate(range(cfg.train.from_pid,
                                    cfg.train.to_pid,
                                    cfg.train.inst_per_step)):
        # Train over n instance
        _to_pid = pid + cfg.train.inst_per_step
        dataloader_cls = dataloader_factory.get(cfg.prob.name)
        dataloader = dataloader_cls(cfg)
        train_dataloader = dataloader.get("train", pid, _to_pid)
        scores = train_step(model,
                            loss_fn,
                            optimizer,
                            train_dataloader,
                            cfg.prob.num_objs,
                            cfg.prob.num_vars,
                            layer_norm_const=layer_norm_const)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

        # Print after each batch of instances is processed
        if verbose:
            acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
            print_result(epoch,
                         "Train",
                         pid=pid,
                         acc=acc,
                         correct=correct,
                         total=total,
                         inst_per_step=cfg.train.inst_per_step)

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


def val_loop(cfg, epoch, model, layer_norm_const=100):
    rng = random.Random(100)
    tp, fp, tn, fn = 0, 0, 0, 0
    scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                             np.zeros((cfg.prob.num_vars, 5))), axis=1)
    scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
    for idx, pid in enumerate(range(cfg.val.from_pid,
                                    cfg.val.to_pid,
                                    cfg.val.inst_per_step)):
        _to_pid = pid + cfg.val.inst_per_step
        dataloader_cls = dataloader_factory.get(cfg.prob.name)
        dataloader = dataloader_cls(cfg)
        val_dataloader = dataloader.get("val", pid, _to_pid)
        scores = val_step(model,
                          val_dataloader,
                          cfg.prob.num_objs,
                          cfg.prob.num_vars,
                          layer_norm_const=layer_norm_const)
        scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)

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
                                  optimizer,
                                  layer_norm_const=cfg.prob.layer_norm_const,
                                  verbose=cfg.train.verbose)
        scores_df, tp, fp, tn, fn = train_result

        # Log training accuracy metrics
        acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
        print_result(epoch,
                     "Train",
                     acc=acc,
                     correct=correct,
                     total=total)
        checkpoint(cfg.train.checkpoint_dir,
                   epoch=epoch,
                   model=model,
                   scores_df=scores_df)

        if (epoch + 1) % cfg.val.every == 0:
            val_result = val_loop(cfg, epoch, model, layer_norm_const=cfg.prob.layer_norm_const)
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
            if is_best:
                checkpoint(cfg.val.checkpoint_dir,
                           epoch=epoch,
                           model=model,
                           scores_df=scores_df)


if __name__ == "__main__":
    main()
