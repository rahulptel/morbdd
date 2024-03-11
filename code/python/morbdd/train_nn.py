import copy
import os
import random
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torchmetrics.classification import StatScores
from morbdd.utils import statscore
from morbdd.model import model_factory
from morbdd.utils import calculate_accuracy
from morbdd.utils import checkpoint
from morbdd.utils import get_context_features
from morbdd.utils import get_split_datasets
from morbdd.utils import print_result
from morbdd.utils import set_device
from morbdd.utils import update_scores
from morbdd import resource_path


# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'


# dataset_dict = {}
# val_dataset = None
# val_dataloader = None


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def init_optimizer(cfg, model):
    opt_cls = getattr(optim, cfg.opt.name)
    opt = opt_cls(model.parameters(), lr=cfg.opt.lr)

    return opt


def init_loss_fn(cfg):
    return nn.BCELoss()


# def train_step(model, loss_fn, opt, dataloader, num_objs, num_vars, device, layer_norm_const=100,
#                flag_layer_penalty=False, flag_label_penalty=False):
#     statscores = StatScores(task='binary', multidim_average='samplewise')
#     scores = None
#
#     num_batches = len(dataloader)
#     # Train over this instance
#     for bid, batch in enumerate(dataloader):
#         sys.stdout.write("\r")
#         sys.stdout.write(f"\t\tProgress: {((bid + 1) / num_batches) * 100:.2f}%")
#         sys.stdout.flush()
#
#         nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
#                                                       batch['wtlayer'], batch['wtlabel'], batch['label'])
#
#         # Get layer ids of the nodes in the current batch and predict
#         lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
#         lidxs = list(map(int, lidxs_t.cpu().numpy()))
#         context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)
#         preds = model(inst_feat, context_feat, nf, pf)
#
#         # Weighted loss
#         weight = None
#         if flag_layer_penalty:
#             weight = wtlayer
#         if flag_label_penalty:
#             weight = wtlabel if weight is None else weight * wtlabel
#         loss_fn = nn.BCELoss(weight=weight.reshape(-1, 1))
#         loss_batch = loss_fn(preds, label)
#
#         opt.zero_grad()
#         loss_batch.backward()
#         opt.step()
#
#         score = statscores(preds.clone().detach(), label)
#         score = score if len(score.shape) > 1 else score.unsqueeze(0)
#         layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
#         scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)
#     print()
#
#     return scores.cpu().numpy()
#
#
# def train_loop(cfg, epoch, model, loss_fn, optimizer, device):
#     print("\tTrain loop...")
#     global dataset_dict
#     tp, fp, tn, fn = 0, 0, 0, 0
#     scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
#                              np.zeros((cfg.prob.num_vars, 5))), axis=1)
#     scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
#     scores_df["layer"] = list(map(int, scores_df["layer"]))
#
#     pids = list(range(cfg.train.from_pid, cfg.train.to_pid))
#     num_pids = len(pids)
#     random.shuffle(pids)
#     idx = 0
#     while idx < num_pids:
#         to_pid = int(np.min([idx + cfg.train.inst_per_step, num_pids]))
#         _pids = pids[idx: to_pid]
#
#         datasets, dataset_dict = get_split_datasets(_pids, cfg.prob.name, cfg.prob.size, "train",
#                                                     cfg.train.neg_pos_ratio,
#                                                     cfg.train.min_samples,
#                                                     dataset_dict,
#                                                     device=device)
#         if len(datasets) == 0:
#             continue
#         dataset = ConcatDataset(datasets)
#         dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
#         print(f"\t\tDataset size: {len(dataset)}")
#         print(f"\t\tNumber of batches: {len(dataloader)}")
#         scores = train_step(model,
#                             loss_fn,
#                             optimizer,
#                             dataloader,
#                             cfg.prob.num_objs,
#                             cfg.prob.num_vars,
#                             device,
#                             layer_norm_const=cfg.prob.layer_norm_const,
#                             flag_layer_penalty=cfg.train.flag_layer_penalty,
#                             flag_label_penalty=cfg.train.flag_label_penalty)
#         scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)
#
#         # Print after each batch of instances is processed
#         if cfg.train.verbose:
#             acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
#             print_result(epoch,
#                          "Train",
#                          pid=idx,
#                          acc=acc,
#                          correct=correct,
#                          total=total,
#                          inst_per_step=cfg.train.inst_per_step)
#
#         idx += cfg.train.inst_per_step
#
#     return scores_df, tp, fp, tn, fn
#
#
# def val_step(model, dataloader, num_objs, num_vars, device, layer_norm_const=100):
#     statscores = StatScores(task='binary', multidim_average='samplewise')
#     model.eval()
#     scores = None
#     num_batches = len(dataloader)
#     with torch.no_grad():
#         for bid, batch in enumerate(dataloader):
#             sys.stdout.write("\r")
#             sys.stdout.write(f"\t\tProgress: {((bid + 1) / num_batches) * 100:.2f}%")
#             sys.stdout.flush()
#
#             nf, pf, inst_feat, wtlayer, wtlabel, label = (batch['nf'], batch['pf'], batch['if'],
#                                                           batch['wtlayer'], batch['wtlabel'], batch['label'])
#
#             # Get layer ids of the nodes in the current batch
#             lidxs_t = torch.round(nf[:, 1] * layer_norm_const)
#             lidxs = list(map(int, lidxs_t.cpu().numpy()))
#             context_feat = get_context_features(lidxs, inst_feat, num_objs, num_vars, device)
#
#             preds = model(inst_feat, context_feat, nf, pf)
#
#             score = statscores(preds.clone().detach(), label)
#             score = score if len(score.shape) > 1 else score.unsqueeze(0)
#             layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
#             scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)
#     print()
#
#     return scores.cpu().numpy()
#
#
# def val_loop(cfg, epoch, model, device):
#     global dataset_dict
#     print("\tValidation loop...")
#     global val_dataloader, val_dataset
#
#     rng = random.Random(100)
#     tp, fp, tn, fn = 0, 0, 0, 0
#     scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
#                              np.zeros((cfg.prob.num_vars, 5))), axis=1)
#     scores_df = pd.DataFrame(scores, columns=["layer", "TP", "FP", "TN", "FN", "Support"])
#
#     pids = list(range(cfg.val.from_pid, cfg.val.to_pid))
#     if val_dataloader is None:
#         datasets, dataset_dict = get_split_datasets(pids, cfg.prob.name, cfg.prob.size, "val",
#                                                     cfg.val.neg_pos_ratio,
#                                                     cfg.val.min_samples,
#                                                     dataset_dict,
#                                                     device)
#         val_dataset = ConcatDataset(datasets)
#         val_dataloader = DataLoader(val_dataset, batch_size=cfg.val.batch_size, shuffle=False)
#
#     print(f"\t\tDataset size: {len(val_dataset)}")
#     print(f"\t\tNumber of batches: {len(val_dataloader)}")
#     scores = val_step(model,
#                       val_dataloader,
#                       cfg.prob.num_objs,
#                       cfg.prob.num_vars,
#                       device,
#                       layer_norm_const=cfg.prob.layer_norm_const)
#     scores_df, tp, fp, tn, fn = update_scores(scores_df, scores, tp, fp, tn, fn)
#
#     return scores_df, tp, fp, tn, fn
#

def aggregate_results(results):
    scores_df, tp, fp, tn, fn = results[0]
    for result in results[1:]:
        _scores_df, _tp, _fp, _tn, _fn = result
        scores_df += _scores_df
        tp += _tp
        fp += _fp
        tn += _tn
        fn += _fn

    return scores_df, tp, fp, tn, fn


def val_worker(cfg, epoch, model, dataloader, device):
    print("\tValidation worker...")
    # statscores = StatScores(task='binary', multidim_average='samplewise')

    tp, fp, tn, fn = 0, 0, 0, 0
    init_scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                                  np.zeros((cfg.prob.num_vars, 4))), axis=1)
    scores_df = pd.DataFrame(init_scores, columns=["layer", "TP", "FP", "TN", "FN"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))
    scores = None

    with torch.no_grad():
        for bidx, batch in enumerate(dataloader):
            nf, pf, inst_feat, label = (batch["nf"], batch["pf"], batch["if"], batch["label"])

            # Get layer ids of the nodes in the current batch and predict
            lidxs_t = torch.round(nf[:, -1] * cfg.prob.layer_norm_const)
            lidxs = list(map(int, lidxs_t.cpu().numpy()))
            context_feat = get_context_features(lidxs, inst_feat, cfg.prob.num_objs, cfg.prob.num_vars, device)
            preds = model(inst_feat, context_feat, nf, pf)

            preds = preds.clone().detach()
            preds = torch.round(preds, decimals=1)
            score = statscore(preds=preds, labels=label, threshold=cfg.threshold, round_upto=cfg.round_upto,
                              is_type="torch")

            score = score if len(score.shape) > 1 else score.unsqueeze(0)
            layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
            scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

        scores = scores.cpu().numpy()
        scores_df, tp, fp, tn, fn = update_scores(cfg.prob.num_vars, scores_df, scores, tp, fp, tn, fn)

    return scores_df, tp, fp, tn, fn


def train_worker(rank, cfg, epoch, model, optimizer, dataloader, device):
    print("\tTrain worker: ", rank, " Batches: ", len(dataloader))
    # statscores = StatScores(task='binary', multidim_average='samplewise')

    tp, fp, tn, fn = 0, 0, 0, 0
    init_scores = np.concatenate((np.arange(cfg.prob.num_vars).reshape(-1, 1),
                                  np.zeros((cfg.prob.num_vars, 4))), axis=1)
    scores_df = pd.DataFrame(init_scores, columns=["layer", "TP", "FP", "TN", "FN"])
    scores_df["layer"] = list(map(int, scores_df["layer"]))
    scores = None

    for bidx, batch in enumerate(dataloader):
        nf, pf, inst_feat, wt, label = (batch["nf"], batch["pf"], batch["if"], batch["wt"], batch["label"])
        # Get layer ids of the nodes in the current batch and predict
        lidxs_t = torch.round(nf[:, -1] * cfg.prob.layer_norm_const)
        lidxs = list(map(int, lidxs_t.cpu().numpy()))
        context_feat = get_context_features(lidxs, inst_feat, cfg.prob.num_objs, cfg.prob.num_vars, device)
        preds = model(inst_feat, context_feat, nf, pf)

        # Weighted loss
        weight = wt \
            if cfg.train.flag_layer_penalty or cfg.train.imbalance_penalty or cfg.train.importance_penalty \
            else None
        loss_fn = nn.BCELoss(weight=weight.reshape(-1, 1))
        loss_batch = loss_fn(preds, label)

        if epoch > 0:
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        preds = preds.clone().detach()
        score = statscore(preds=preds, labels=label, threshold=cfg.threshold, round_upto=cfg.round_upto,
                          is_type="torch")
        score = score if len(score.shape) > 1 else score.unsqueeze(0)
        layer_score = torch.cat((lidxs_t.unsqueeze(1), score), axis=1)
        scores = layer_score if scores is None else torch.cat((scores, layer_score), axis=0)

    scores = scores.cpu().numpy()
    scores_df, tp, fp, tn, fn = update_scores(cfg.prob.num_vars, scores_df, scores, tp, fp, tn, fn)
    return scores_df, tp, fp, tn, fn


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_nn.yaml")
def main(cfg):
    torch.set_num_threads(1)
    device = set_device(cfg.device)

    model_cls = model_factory.get("ParetoStatePredictor")
    model = model_cls(cfg.mdl)
    model.to(device)
    model.share_memory()
    optimizer = SharedAdam(model.parameters())

    sampling_type = "npr1ms0"
    labels_type = "binary"
    weights_type = "exponential-0-1-sum"

    print("Building training dataset...")
    start = time.time()
    train_pids = list(range(cfg.train.from_pid, cfg.train.to_pid))
    train_datasets, _ = get_split_datasets(train_pids,
                                           cfg.prob.name,
                                           cfg.prob.size,
                                           "train",
                                           sampling_type,
                                           labels_type,
                                           weights_type,
                                           device)
    train_dataset = ConcatDataset(train_datasets)
    end = time.time()
    print(f"Took {end - start:.2f}s to build the train dataset...")
    start = time.time()
    val_pids = list(range(cfg.val.from_pid, cfg.val.to_pid))
    val_datasets, _ = get_split_datasets(val_pids,
                                         cfg.prob.name,
                                         cfg.prob.size,
                                         "val",
                                         sampling_type,
                                         labels_type,
                                         weights_type,
                                         device)
    val_dataset = ConcatDataset(val_datasets)
    end = time.time()
    print(f"Took {end - start:.2f}s to build the validation dataset...")

    best_acc = 0
    for epoch in range(cfg.train.epochs):
        ep_start = time.time()
        print(f"Epoch {epoch}")

        model.train()
        pool = mp.Pool(processes=cfg.train.num_processes)
        results = []
        for rank in range(cfg.train.num_processes):
            dataloader = DataLoader(train_dataset,
                                    batch_size=cfg.train.batch_size,
                                    sampler=DistributedSampler(train_dataset,
                                                               num_replicas=cfg.train.num_processes,
                                                               rank=rank))
            results.append(pool.apply_async(train_worker,
                                            (rank, cfg, epoch, model, optimizer, dataloader, device)))
        train_results = [p.get() for p in results]
        # dataloader = DataLoader(train_dataset,
        #                         batch_size=cfg.train.batch_size,
        #                         shuffle=True)
        # train_results = train_worker(0, cfg, epoch, model, optimizer, dataloader, device)
        train_end = time.time()
        print("\t\tTraining time ", train_end - ep_start)
        scores_df, tp, fp, tn, fn = aggregate_results(train_results)
        # scores_df, tp, fp, tn, fn = aggregate_results([train_results])

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

        if (epoch + 1) % cfg.val.every == 1:
            val_start = time.time()
            model.eval()
            pool = mp.Pool(processes=cfg.train.num_processes)
            results = []
            for rank in range(cfg.train.num_processes):
                dataloader = DataLoader(val_dataset,
                                        batch_size=cfg.val.batch_size,
                                        sampler=DistributedSampler(val_dataset,
                                                                   num_replicas=cfg.train.num_processes,
                                                                   rank=rank))
                results.append(pool.apply_async(val_worker,
                                                (cfg, epoch, model, dataloader, device)))
            val_results = [p.get() for p in results]
            # dataloader = DataLoader(val_dataset,
            #                         batch_size=cfg.train.batch_size,
            #                         shuffle=True)
            # val_results = val_worker(cfg, epoch, model, dataloader, device)
            val_end = time.time()
            print("\t\tValidation time ", val_end - val_start)
            scores_df, tp, fp, tn, fn = aggregate_results(val_results)
            # scores_df, tp, fp, tn, fn = aggregate_results([val_results])

            acc, correct, total = calculate_accuracy(tp, fp, tn, fn)
            is_best = acc > best_acc
            if is_best:
                best_acc = copy.copy(acc)
            # Log validation accuracy metrics
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
