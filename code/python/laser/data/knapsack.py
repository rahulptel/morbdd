import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import json
from operator import itemgetter
import zipfile

from laser import resource_path

ZERO_ARC = -1
ONE_ARC = 1


class BDDDataset(Dataset):
    def __init__(self, node_feat, parent_feat, inst_feat, wt_layer, wt_label, labels):
        self.node_feat = torch.from_numpy(np.array(node_feat)).float()
        self.parent_feat = torch.from_numpy(np.array(parent_feat)).float()
        self.inst_feat = torch.from_numpy(inst_feat).float()
        self.inst_feat = self.inst_feat.T
        self.wt_layer = torch.from_numpy(np.array(wt_layer)).float()
        self.wt_label = torch.from_numpy(np.array(wt_label)).float()
        self.labels = torch.from_numpy(np.array(labels)).float().unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'nf': self.node_feat[i],
                'pf': self.parent_feat[i],
                'if': self.inst_feat,
                'wtlayer': self.wt_layer[i],
                'wtlabel': self.wt_label[i],
                'label': self.labels[i]}


def get_parent_features(node, bdd, lidx, state_norm_const):
    parents_feat = []

    for p in node['op']:
        parent_state = 0 if lidx == 0 else bdd[lidx - 1][p]['s'][0] / state_norm_const
        parents_feat.append([ONE_ARC, parent_state])

    for p in node['zp']:
        # Parent state will be the same as the current state for zero-arc
        parent_state = node['s'][0] / state_norm_const
        parents_feat.append([ZERO_ARC, parent_state])

    return parents_feat


def read_instance(archive, inst):
    data = {'value': [], 'n_vars': 0, 'n_cons': 1, 'n_objs': 3}
    data['weight'], data['capacity'] = [], 0

    zf = zipfile.ZipFile(archive)
    raw_data = zf.open(inst, 'r')
    data['n_vars'] = int(raw_data.readline())
    data['n_objs'] = int(raw_data.readline())
    for _ in range(data['n_objs']):
        data['value'].append(list(map(int, raw_data.readline().split())))
    data['weight'].extend(list(map(int, raw_data.readline().split())))
    data['capacity'] = int(raw_data.readline().split()[0])

    return data


def get_instance_data(problem, size, split, pid):
    # TODO: At the moment the all the instances should be mapped to 1000-1009 of the validation set.
    #  It should be removed this as soon as possible.
    if problem == 'knapsack':
        prefix = 'kp_7'

    archive = resource_path / f"instances/{problem}/{size}.zip"
    # archive = ROOTPATH / f'resources/instances/knapsack/{size}.zip'
    # TODO: Change this to the respective split instead of val
    # inst = f'{size}/{split}/{prefix}_{size}_{pid}.dat'
    pid_val = (pid % 10) + 1000
    inst = f'{size}/val/{prefix}_{size}_{pid_val}.dat'
    data = read_instance(archive, inst)
    return data


def get_bdd_data(problem, size, split, pid):
    archive = resource_path / f"bdds/{problem}/{size}.zip"
    zf = zipfile.ZipFile(archive)
    fp = zf.open(f"{size}/{split}/{pid}.json", "r")
    bdd = json.load(fp)

    return bdd


def get_instance_features(problem, data, state_norm_const=None):
    def get_knapsack_instance_features():
        _feat = np.concatenate((np.array(data['value']), np.array(data['weight']).reshape(1, -1)), axis=0)

        assert state_norm_const is not None
        _feat = _feat / state_norm_const

        return _feat

    feat = None
    if problem == "knapsack":
        feat = get_knapsack_instance_features()

    assert feat is not None
    return feat


def get_order(problem, order_type, data):
    def get_knapsack_order(order_type, data):
        if order_type == 'MinWt':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1))

            return np.array([i[0] for i in idx_weight])
        elif order_type == 'MaxRatio':
            min_profit = np.min(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(min_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)

            return np.array([i[0] for i in idx_profit_by_weight])
        elif order_type == 'Lex':
            return np.arange(data['n_vars'])

    order = None
    if problem == 'knapsack':
        order = get_knapsack_order(order_type, data)

    assert order is not None

    return order


class BDDDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def _get_knapsack_bdd_dataset(self,
                                  split="val",
                                  pid=1000):
        rng = random.Random(100)
        # Load bdd
        bdd = get_bdd_data(self.cfg.prob.name, self.cfg.prob.size, split, pid)
        lidxs = [lidx / self.cfg.prob.num_vars for lidx in range(self.cfg.prob.num_vars)]

        # Set layer weights
        if self.cfg.train.layer_penalty == "const":
            layer_weight = [1 for _ in self.cfg.prob.num_vars]
        elif self.cfg.train.layer_penalty == "linear":
            layer_weight = [1 - lidx for lidx in lidxs]
        elif self.cfg.train.layer_penalty == "exponential":
            layer_weight = [np.exp(-0.5 * lidx) for lidx in lidxs]
        elif self.cfg.prob.wt_layer == "linearE":
            layer_weight = [(np.exp(-0.5) - 1) * lidx + 1
                            for lidx in lidxs]
        elif self.cfg.prob.wt_layer == "quadratic":
            layer_weight = [(np.exp(-0.5) - 1) * (lidx ** 2) + 1
                            for lidx in lidxs]
        elif self.cfg.prob.wt_layer == "sigmoidal":
            layer_weight = [(1 + np.exp(-0.5)) / (1 + np.exp(lidx - 0.5))
                            for lidx in lidxs]
        else:
            raise ValueError("Invalid layer penalty scheme!")

        # Get instance features
        data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, split, pid)
        order = get_order(self.cfg.prob.name, self.cfg.prob.order, data)
        inst_feat = get_instance_features(self.cfg.prob.name,
                                          data,
                                          state_norm_const=self.cfg.prob.state_norm_const)
        # Reset order of the instance
        inst_feat = inst_feat[:, order]

        # Get node and parent features
        node_feat, parents_feat = [], []
        wt_layer, wt_label = [], []
        labels = []
        for lidx, layer in enumerate(bdd):
            if lidx > self.cfg[split].dataset_upto_layer:
                break
            _node_feat, _parents_feat = [], []
            _wt_layer, _wt_label = [], []
            _labels = []
            num_pos = np.sum([1 for node in layer if node['l'] == 1])
            for nidx, node in enumerate(layer):
                # Append to global
                if node['l'] == 1:
                    # Extract node feature
                    node_feat.append([node['s'][0] / self.cfg.prob.state_norm_const,
                                      (lidx + 1) / self.cfg.prob.layer_norm_const])
                    # Extract parent feature
                    parents_feat.append(get_parent_features(node,
                                                            bdd,
                                                            lidx,
                                                            self.cfg.prob.state_norm_const))
                    wt_layer.append(layer_weight[lidx])
                    wt_label.append(1)
                    labels.append(1)
                    # Append to local
                else:
                    # Extract node feature
                    _node_feat.append([node['s'][0] / self.cfg.prob.state_norm_const,
                                       (lidx + 1) / self.cfg.prob.layer_norm_const])
                    # Extract parent feature
                    _parents_feat.append(get_parent_features(node,
                                                             bdd,
                                                             lidx,
                                                             self.cfg.prob.state_norm_const))
                    _wt_layer.append(layer_weight[lidx])
                    _wt_label.append(1 / self.cfg[split].neg_pos_ratio)
                    _labels.append(0)

            # Select samples from local to append to the global
            if self.cfg[split].neg_pos_ratio == -1:
                # Select all negative samples
                node_feat.extend(_node_feat)
                parents_feat.extend(_parents_feat)
                wt_layer.extend(_wt_layer)
                wt_label.extend(_wt_label)
                labels.extend(_labels)
            else:
                neg_idxs = list(range(len(_labels)))
                rng.shuffle(neg_idxs)
                num_neg_samples = len(_labels) if len(_labels) < num_pos else num_pos
                for nidx in neg_idxs[:num_neg_samples]:
                    node_feat.append(_node_feat[nidx])
                    parents_feat.append(_parents_feat[nidx])
                    wt_layer.append(_wt_layer[nidx])
                    wt_label.append(_wt_label[nidx])
                    labels.append(_labels[nidx])

        # Pad parents
        max_parents = np.max([len(pf) for pf in parents_feat])

        parents_feat_padded = []
        for pf in parents_feat:
            if len(pf) < max_parents:
                parents_feat_padded.append(np.concatenate((pf, np.zeros((max_parents - len(pf), len(pf[0]))))))
            else:
                parents_feat_padded.append(pf)

        return BDDDataset(node_feat, parents_feat_padded, inst_feat, wt_layer, wt_label, labels)

    def _get_bdd_dataset(self, split="train", pid=0):
        dataset = None
        if self.cfg.prob.name == "knapsack":
            dataset = self._get_knapsack_bdd_dataset(split=split, pid=pid)

        assert dataset is not None

        return dataset

    def get(self,
            split,
            from_pid,
            to_pid):

        datasets = []
        for pid in range(from_pid, to_pid):
            # Get dataset and dataloader for the instance
            dataset = self._get_bdd_dataset(split=split, pid=pid)
            datasets.append(dataset)

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg[split].batch_size,
                                shuffle=self.cfg[split].shuffle)

        return dataloader
