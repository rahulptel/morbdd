import hashlib
import io
import json
import os
import random
import zipfile
from operator import itemgetter

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader

from morbdd import CONST
from morbdd import ResourcePaths as path


def zipdir(path, ziph):
    # Iterate over all the files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Create the relative path to maintain the folder structure
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def get_env(n_objs=3):
    modname = "libbddenvv2o" + str(n_objs)
    libbddenv = __import__(modname)
    env = libbddenv.BDDEnv()

    return env


def get_dataset_prefix(with_parent=False, layer_weight=None, neg_to_pos_ratio=1.0):
    prefix = []
    if with_parent:
        prefix.append("wp")
    if layer_weight is not None:
        prefix.append(f"{layer_weight}")
    if neg_to_pos_ratio != 1.0:
        prefix.append(f"{neg_to_pos_ratio}")

    if len(prefix):
        prefix = "-".join(prefix)
    else:
        prefix = "default"

    return prefix


def get_dataset_path(cfg):
    file_path = path.dataset / f"{cfg.prob.name}/{cfg.prob.size}/{cfg.split}"
    prefix = get_dataset_prefix(cfg.with_parent, cfg.layer_weight, cfg.neg_to_pos_ratio)
    file_path /= prefix

    return file_path


class Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingHelper:
    def __init__(self):
        self.train_stats = []
        self.val_stats = []

    @staticmethod
    def compute_batch_stats(labels, preds):
        # True positive: label=1 and class=1
        tp = (preds[labels == 1] == 1).sum()
        # True negative: label=0 and class=0
        tn = (preds[labels == 0] == 0).sum()
        # False positive: label=0 and class=1
        fp = (preds[labels == 0] == 1).sum()
        # False negative: label=1 and class=0
        fn = (preds[labels == 1] == 0).sum()

        pos = labels.sum()
        neg = labels.shape[0] - pos
        # items = pos + neg
        return tp, tn, fp, fn, pos, neg

    @staticmethod
    def compute_meta_stats(stats, prefix=""):
        loss, tp, tn, fp, fn, n_pos, n_neg = (stats[prefix + "loss"], stats[prefix + "tp"], stats[prefix + "tn"],
                                              stats[prefix + "fp"], stats[prefix + "fn"], stats[prefix + "n_pos"],
                                              stats[prefix + "n_neg"])

        loss = loss / (n_pos + n_neg)
        acc = ((tp + tn) / (tp + fp + tn + fn))
        f1 = tp / (tp + (0.5 * (fn + fp)))
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)

        meta_stats = {
            prefix + "loss": loss, prefix + "acc": acc, prefix + "f1": f1, prefix + "precision": precision,
            prefix + "recall": recall, prefix + "specificity": specificity
        }

        return meta_stats

    @staticmethod
    def print_batch_stats(epoch, batch_id, stats):
        print("EP-{}:{}: Batch loss: {:.3f}, Acc: {:.3f}, F1: {:.3f}, "
              "Precision:{:.3f}, Recall:{:.3f}, Specificity:{:.3f}, "
              "Time: {:.2f}, Items: {}".format(epoch,
                                               batch_id,
                                               stats["loss"],
                                               stats["acc"],
                                               stats["f1"],
                                               stats["precision"],
                                               stats["recall"],
                                               stats["specificity"],
                                               stats["time"],
                                               stats["items"]))

    @staticmethod
    def print_stats(split, stats, prefix=""):
        epoch = stats["epoch"]
        ept, bt, dt, = stats[prefix + "epoch_time"], stats[prefix + "batch_time"], stats[prefix + "data_time"]

        print_str = ("{}:{}: F1: {:4f}, Acc: {:.4f}, Loss {:.4f}, Recall: {:.4f}, Precision: {:.4f}, "
                     "Specificity: {:.4f}, Epoch Time: {:.4f}, Batch Time: {:.4f}, Data Time: {:.4f}")
        print(print_str.format(epoch, prefix + split, stats[prefix + "f1"], stats[prefix + "acc"],
                               stats[prefix + "loss"], stats[prefix + "recall"], stats[prefix + "precision"],
                               stats[prefix + "specificity"], ept, bt, dt))

        print_str = "{}: Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f},"
        for i in ["0", "1"]:
            print(print_str.format("lgt" + i,
                                   stats[prefix + "lgt" + i + "-mean"],
                                   stats[prefix + "lgt" + i + "-std"],
                                   stats[prefix + "lgt" + i + "-min"],
                                   stats[prefix + "lgt" + i + "-max"]))
        print("--------------------------")

    # def save(self, epoch, save_path, best_model=False, model=None, optimizer=None):
    #     print("Saving model={}".format(best_model))
    #     model_path = save_path / "model.pt"
    #     print("Saving model to: {}".format(model_path))
    #     model_obj = {"epoch": epoch, "model": model, "optimizer": optimizer}
    #     torch.save(model_obj, model_path)
    #     if best_model:
    #         model_path = save_path / "best_model.pt"
    #         torch.save(model_obj, model_path)

    @staticmethod
    def save_model_and_opt(epoch, save_path, best_model=False, model=None, optimizer=None):
        # print(epoch)
        # print("Is best: {}".format(best_model))
        if best_model:
            model_path = save_path / "best_model.pt"
        else:
            model_path = save_path / "model.pt"
        # print("Saving model to: {}".format(model_path))

        model_obj = {"epoch": epoch + 1, "model": model, "optimizer": optimizer}
        torch.save(model_obj, model_path)

    def save_stats(self, save_path):
        stats_path = save_path / f"stats.pt"
        # print("Saving stats to: {}".format(stats_path))
        stats_obj = {"train": self.train_stats, "val": self.val_stats}
        torch.save(stats_obj, stats_path)

    def compute_meta_stats_and_print(self, split, stats, prefix=""):
        meta_stats = self.compute_meta_stats(stats, prefix=prefix)
        stats.update(meta_stats)
        self.print_stats(split, stats, prefix=prefix)

        return stats


class KnapsackBDDDataset(Dataset):
    def __init__(self, size=None, split=None, pid=None,
                 sampling_type=None, labels_type=None, weights_type=None, device=None):
        super(KnapsackBDDDataset, self).__init__()

        zf = zipfile.ZipFile(path.resource / f"tensors/knapsack/{size}/{split}/{sampling_type}.zip")
        self.node_feat = torch.load(zf.open(f"{sampling_type}/n{pid}.pt")).to(device)
        self.parent_feat = torch.load(zf.open(f"{sampling_type}/p{pid}.pt")).to(device)
        self.inst_feat = torch.load(zf.open(f"{sampling_type}/i{pid}.pt")).to(device)
        self.wt = torch.load(zf.open(f"{sampling_type}/{weights_type}/{pid}.pt")).to(device)

        zf = zipfile.ZipFile(path.resource / f"tensors/knapsack/{size}/{split}/labels/{labels_type}.zip")
        self.labels = torch.load(zf.open(f"{labels_type}/{pid}.pt")).to(device)

        # self.node_feat = data["nf"].to(device)
        # self.parent_feat = data["pf"].to(device)
        # self.inst_feat = data["if"].to(device)
        # self.wt_layer = data["wtlayer"].to(device)
        # self.wt_label = data["wtlabel"].to(device)
        # self.labels = data["label"].to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'nf': self.node_feat[i],
                'pf': self.parent_feat[i],
                'if': self.inst_feat,
                'wt': self.wt[i],
                'label': self.labels[i]}


class FeaturizerConfig:
    def __init__(self, norm_const=1000, raw=False, context=True):
        self.norm_const = norm_const
        self.raw = raw
        self.context = context


def read_from_zip(archive, file, format="raw"):
    try:
        zf = zipfile.ZipFile(archive)
        raw_data = zf.open(file, "r")
    except:
        raw_data = None

    data = None
    if raw_data is not None:
        if format == "raw":
            data = raw_data
        elif format == "json":
            data = json.load(raw_data)
        elif format == "npz":
            data = np.load(io.BytesIO(raw_data.read()))
        elif format == "pt":
            data = torch.load(io.BytesIO(raw_data.read()))

    return data


def read_instance_knapsack(archive, inst):
    data = {'value': [], 'n_vars': 0, 'n_cons': 1, 'n_objs': 3}
    data['weight'], data['capacity'] = [], 0

    raw_data = read_from_zip(archive, inst, format="raw")
    data['n_vars'] = int(raw_data.readline())
    data['n_objs'] = int(raw_data.readline())
    for _ in range(data['n_objs']):
        data['value'].append(list(map(int, raw_data.readline().split())))
    data['weight'].extend(list(map(int, raw_data.readline().split())))
    data['capacity'] = int(raw_data.readline().split()[0])

    return data


def read_instance_indepset(archive, inst):
    if inst.split(".")[-1] == "npz":
        data = read_from_zip(archive, inst, format="npz")
    else:
        raw_data = read_from_zip(archive, inst)

        data = {"obj_coeffs": [], "cons_coeffs": [], "rhs": []}

        data["n_vars"], data["n_cons"] = list(map(int, raw_data.readline().strip().split()))
        data["n_objs"] = int(raw_data.readline())
        data["adj_list"] = np.zeros((data["n_vars"], data["n_vars"]))
        data["adj_list_comp"] = np.ones((data["n_vars"], data["n_vars"]))
        for i in range(data["n_vars"]):
            data["adj_list"][i, i] = 1
            data["adj_list_comp"][i, i] = 0

        for _ in range(data["n_objs"]):
            data["obj_coeffs"].append(list(map(int, raw_data.readline().split())))
        # print(data["obj_coeffs"])

        for _ in range(data["n_cons"]):
            n_vars_per_con = list(map(int, raw_data.readline().strip().split()))[0]
            non_zero_vars = list(map(int, raw_data.readline().strip().split()))
            # print(n_vars_per_con, non_zero_vars)
            non_zero_vars = [i - 1 for i in non_zero_vars]
            data["cons_coeffs"].append(non_zero_vars)

            for i in range(len(non_zero_vars)):
                i_var = non_zero_vars[i]
                for j in range(i + 1, len(non_zero_vars)):
                    j_var = non_zero_vars[j]
                    data["adj_list"][i_var, j_var] = 1
                    data["adj_list"][j_var, i_var] = 1
                    data["adj_list_comp"][i_var, j_var] = 0
                    data["adj_list_comp"][j_var, i_var] = 0

        # print(data["adj_list"])
        # data["adj_list_comp"] = np.zeros((data["n_vars"], data["n_vars"]))
        # data["adj_list_comp"][data["adj_list"] == 0] = 1
        # print(data["adj_list"][96])
        # print(data["adj_list_comp"][96])
    return data


def read_instance(problem, archive, inst):
    data = None
    if problem == "knapsack" or problem == "knapsackc":
        data = read_instance_knapsack(archive, inst)
    elif problem == "indepset":
        data = read_instance_indepset(archive, inst)
    return data


def get_instance_prefix(problem):
    prefix = None
    if problem == 'knapsack' or problem == 'knapsackc':
        prefix = 'kp_7'
    elif problem == "indepset":
        prefix = "ind_7"

    return prefix


def get_instance_data(problem, size, split, pid):
    prefix = get_instance_prefix(problem)
    archive = path.inst / f"{problem}/{size}.zip"
    suffix = "dat"
    if problem == "indepset":
        if len(size.split("-")) > 2:
            suffix = "npz"

    inst = f'{size}/{split}/{prefix}_{size}_{pid}.{suffix}'
    data = read_instance(problem, archive, inst)

    return data


def get_context_features(layer_idxs, inst_feat, num_objs, num_vars, device):
    max_lidx = np.max(layer_idxs)
    context = []
    for inst_idx, lidx in enumerate(layer_idxs):
        _inst_feat = inst_feat[inst_idx, :lidx, :]

        ranks = (torch.arange(lidx).reshape(-1, 1) + 1) / num_vars
        _context = torch.concat((_inst_feat, ranks.to(device)), axis=1)

        pad = torch.zeros(max_lidx - _inst_feat.shape[0], num_objs + 2).to(device)
        _context = torch.concat((_context, pad), axis=0)

        context.append(_context)
    context = torch.stack(context)

    return context


def get_layer_weights_const(num_vars):
    return [1 for _ in range(num_vars)]


def get_layer_weights_linear(lidxs):
    return [1 - lidx for lidx in lidxs]


def get_layer_weights_exponential(lidxs):
    return [np.exp(-0.5 * lidx) for lidx in lidxs]


def get_layer_weights_linearE(lidxs):
    return [(np.exp(-0.5) - 1) * lidx + 1 for lidx in lidxs]


def get_layer_weights_quadratic(lidxs):
    return [(np.exp(-0.5) - 1) * (lidx ** 2) + 1 for lidx in lidxs]


def get_layer_weights_sigmoidal(lidxs):
    return [(1 + np.exp(-0.5)) / (1 + np.exp(lidx - 0.5)) for lidx in lidxs]


def get_layer_weights(flag_penalty, penalty, num_vars):
    lidxs = [lidx / num_vars for lidx in range(num_vars)]
    get_layer_weights_fn = {
        "const": get_layer_weights_const,
        "linear": get_layer_weights_linear,
        "exponential": get_layer_weights_exponential,
        "linearE": get_layer_weights_linearE,
        "sigmoidal": get_layer_weights_sigmoidal
    }
    if flag_penalty is False or penalty == "const":
        return get_layer_weights_fn["const"](num_vars)

    if "+" not in penalty:
        layer_weight = get_layer_weights_fn[penalty](lidxs)
    else:
        a, b = penalty.strip().split("+")
        a1, a2 = a.split("-")
        upto_layer = int(a2)
        if a1 == "const":
            layer_weight_a = get_layer_weights_fn["const"](upto_layer)
        else:
            layer_weight_a = get_layer_weights_fn[a1](lidxs[:upto_layer])

        if b == "const":
            layer_weight_b = get_layer_weights_fn["const"](num_vars)
        else:
            layer_weight_b = get_layer_weights_fn[b](lidxs)

        layer_weight = layer_weight_a
        layer_weight.extend(layer_weight_b[:num_vars - upto_layer])

    return layer_weight


def get_bdd_data(problem, size, split, pid):
    archive = path.bdd / f"{problem}/{size}.zip"
    zf = zipfile.ZipFile(archive)
    fp = zf.open(f"{size}/{split}/{pid}.json", "r")
    bdd = json.load(fp)

    return bdd


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


def get_static_order(problem, order_type, data):
    order = None
    if problem == 'knapsack' or problem == 'knapsackc':
        order = get_knapsack_order(order_type, data)
    elif problem == 'indepset':
        order = []
    assert order is not None

    return order


def get_featurizer(problem, cfg):
    featurizer = None
    if problem == "knapsack" or problem == "knapsackc":
        from morbdd.featurizer import KnapsackFeaturizer

        featurizer = KnapsackFeaturizer(cfg)

    assert featurizer is not None

    return featurizer


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


def get_parent_features(problem, node, bdd, lidx, inst_data, state_norm_const):
    def get_parent_features_knapsack():
        parents_feat = []

        for p in node['op']:
            parents_feat.append([
                CONST.ONE_ARC,
                0 if lidx == 0 else bdd[lidx - 1][p]['s'][0] / state_norm_const,
                0 if lidx == 0 else bdd[lidx - 1][p]['s'][0] / inst_data['capacity'],
            ])

        for _ in node['zp']:
            # Parent state will be the same as the current state for zero-arc
            parents_feat.append([
                CONST.ZERO_ARC,
                node['s'][0] / state_norm_const,
                node['s'][0] / inst_data['capacity'],
            ])

        return parents_feat

    parent_features = None
    if problem == "knapsack":
        parent_features = get_parent_features_knapsack()

    assert parent_features is not None

    return parent_features


def np2tensor(data):
    return torch.from_numpy(data).float()


def convert_bdd_to_tensor_data(problem,
                               bdd=None,
                               num_objs=None,
                               num_vars=None,
                               split=None,
                               pid=None,
                               order_type=None,
                               state_norm_const=1000,
                               layer_norm_const=100,
                               task="classification",
                               label_type="binary",
                               neg_pos_ratio=1,
                               min_samples=0,
                               flag_layer_penalty=True,
                               layer_penalty=None,
                               flag_imbalance_penalty=None,
                               flag_importance_penalty=None,
                               penalty_aggregation="sum",
                               random_seed=100):
    size = f"{num_objs}_{num_vars}"

    sampling_type = f"npr{neg_pos_ratio}ms{min_samples}"
    sampling_data_path = path.resource / "tensors" / problem / size / split / sampling_type
    sampling_data_path.mkdir(parents=True, exist_ok=True)
    features_exists = sampling_data_path.joinpath(f"{pid}.pt").exists()
    features_exists = False

    labels_data_path = path.resource / "tensors" / problem / size / split / "labels" / label_type
    labels_data_path.mkdir(parents=True, exist_ok=True)
    labels_exists = labels_data_path.joinpath(f"{pid}.pt").exists()
    labels_exists = False

    weights_type = ""
    if flag_layer_penalty:
        weights_type += f"{layer_penalty}-"
    weights_type += "1-" if flag_imbalance_penalty else "0-"
    weights_type += "1-" if flag_importance_penalty else "0-"
    weights_type += penalty_aggregation
    weights_data_path = path.resource / "tensors" / problem / size / split / sampling_type / weights_type
    weights_data_path.mkdir(parents=True, exist_ok=True)
    weights_exists = weights_data_path.joinpath(f"{pid}.pt").exists()
    weights_exists = False

    print(f"Processed {pid}, Features - {features_exists}, Weights - {weights_exists}, Labels - {labels_exists}")

    def convert_bdd_to_tensor_dataset_knapsack():
        rng = random.Random(random_seed)
        # features_lst = None if features_exists else []
        node_feat_lst = None if features_exists else []
        parents_node_feat_lst = None if features_exists else []
        weights_lst = None if weights_exists else []
        labels_lst = None if labels_exists else []
        layer_weight = get_layer_weights(flag_layer_penalty, layer_penalty, num_vars)

        node_feat, parents_node_feat = None, None
        labels = None
        weights = None

        # Load bdd
        _bdd = get_bdd_data(problem, size, split, pid) if bdd is None else bdd

        # Get instance features
        inst_data = get_instance_data(problem, size, split, pid)
        order = get_static_order(problem, order_type, inst_data)
        inst_feat = get_instance_features(problem,
                                          inst_data,
                                          state_norm_const=state_norm_const)
        # Reset order of the instance
        inst_feat = inst_feat[:, order]

        for lidx, layer in enumerate(_bdd):
            pos_ids = [node_id for node_id, node in enumerate(layer) if node["pareto"] == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))

            # Subsample negative samples
            num_pos_samples = len(pos_ids)
            if neg_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = np.max([int(neg_pos_ratio * num_pos_samples),
                                          min_samples])
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            # Imbalance weights for the current layer
            pos_imb_wt = num_neg_samples / (num_pos_samples + num_neg_samples)
            neg_imb_wt = 1 - pos_imb_wt

            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for i, node_id in enumerate(node_ids):
                node = layer[node_id]

                if not features_exists:
                    # Extract node feature
                    node_feat_lst.append([node['s'][0] / state_norm_const,
                                          node['s'][0] / inst_data['capacity'],
                                          (lidx + 1) / layer_norm_const])
                    # print(node_feat_lst)

                    # Extract parent feature
                    parents_node_feat_lst.append(get_parent_features(problem,
                                                                     node,
                                                                     _bdd,
                                                                     lidx,
                                                                     inst_data,
                                                                     state_norm_const))

                if not weights_exists:
                    weights_lst.append(get_aggregated_weight(
                        aggregation=penalty_aggregation,
                        flag_layer_penalty=flag_layer_penalty,
                        layer_weight=layer_weight[lidx],
                        flag_imbalance_penalty=flag_imbalance_penalty,
                        imb_wt=pos_imb_wt if i < num_pos_samples else neg_imb_wt,
                        flag_importance_penalty=flag_importance_penalty if i < num_pos_samples else False,
                        score=node["score"]))

                if not labels_exists:
                    labels_lst.append(node["l"])

        if not features_exists:
            # Pad parents
            max_parents = np.max([len(pf) for pf in parents_node_feat_lst])
            parents_node_feat_padded = []
            for pf in parents_node_feat_lst:
                if len(pf) < max_parents:
                    parents_node_feat_padded.append(np.concatenate((pf, np.zeros((max_parents - len(pf), len(pf[0]))))))
                else:
                    parents_node_feat_padded.append(pf)

            node_feat = np2tensor(np.array(node_feat_lst))
            parents_node_feat = np2tensor(np.array(parents_node_feat_padded))
            inst_feat = np2tensor(np.array(inst_feat).T)

        if not labels_exists:
            labels = np2tensor(np.array(labels_lst).reshape(-1, 1))

        if not weights_exists:
            weights = np2tensor(np.array(weights_lst))

        return inst_feat, node_feat, parents_node_feat, labels, weights

    if problem == "knapsack":
        data = convert_bdd_to_tensor_dataset_knapsack()
    else:
        raise ValueError("Invalid problem type!")

    inst_feat, node_feat, parents_node_feat, labels, weights = data
    if node_feat is not None:
        torch.save(node_feat, sampling_data_path.joinpath(f"n{pid}.pt"))
        torch.save(parents_node_feat, sampling_data_path.joinpath(f"p{pid}.pt"))
        torch.save(inst_feat, sampling_data_path.joinpath(f"i{pid}.pt"))
    if labels is not None:
        torch.save(labels, labels_data_path.joinpath(f"{pid}.pt"))
    if weights is not None:
        torch.save(weights, weights_data_path.joinpath(f"{pid}.pt"))


def extract_node_features(problem,
                          lidx,
                          node,
                          prev_layer,
                          inst_data,
                          layer_norm_const=None,
                          state_norm_const=None):
    def extract_node_features_knapsack():
        # Node features
        norm_state = node["s"][0] / state_norm_const
        state_to_capacity = node["s"][0] / inst_data["capacity"]
        _node_feat = np.array([norm_state, state_to_capacity, (lidx + 1) / layer_norm_const])

        # Parent node features
        _parent_node_feat = []
        if lidx == 0:
            _parent_node_feat.extend([1, -1, -1, -1, -1, -1])
        else:
            # 1 implies parent of the one arc
            _parent_node_feat.append(1)
            if len(node["op"]) > 1:
                prev_node_idx = node["op"][0]
                prev_state = prev_layer[prev_node_idx]["s"][0]
                _parent_node_feat.append(prev_state / state_norm_const)
                _parent_node_feat.append(prev_state / inst_data["capacity"])
            else:
                _parent_node_feat.append(-1)
                _parent_node_feat.append(-1)

            # -1 implies parent of the zero arc
            _parent_node_feat.append(-1)
            if len(node["zp"]) > 0:
                _parent_node_feat.append(norm_state)
                _parent_node_feat.append(state_to_capacity)
            else:
                _parent_node_feat.append(-1)
                _parent_node_feat.append(-1)
        _parent_node_feat = np.array(_parent_node_feat)

        return _node_feat, _parent_node_feat

    if problem == "knapsack":
        node_feat, parent_node_feat = extract_node_features_knapsack()
        return node_feat, parent_node_feat
    else:
        raise ValueError("Invalid problem!")


def get_aggregated_weight(aggregation="sum",
                          flag_layer_penalty=False,
                          layer_weight=1,
                          flag_imbalance_penalty=False,
                          imb_wt=1,
                          flag_importance_penalty=False,
                          score=0):
    weight = None
    if aggregation == "sum":
        l_wt = layer_weight if flag_layer_penalty else 0
        imb_wt = imb_wt if flag_imbalance_penalty else 0
        imp_wt = score if flag_importance_penalty else 0
        weight = l_wt + imb_wt + imp_wt
    elif aggregation == "mul":
        l_wt = layer_weight if flag_layer_penalty else 1
        imb_wt = imb_wt if flag_imbalance_penalty else 1
        imp_wt = score if flag_importance_penalty else 1
        weight = l_wt * imb_wt * imp_wt

    if weight is None or weight == 0:
        return 1

    return weight


def convert_bdd_to_xgb_data(problem,
                            bdd=None,
                            num_objs=None,
                            num_vars=None,
                            split=None,
                            pid=None,
                            order_type=None,
                            state_norm_const=1000,
                            layer_norm_const=100,
                            task="classification",
                            label_type="binary",
                            neg_pos_ratio=1,
                            min_samples=0,
                            flag_layer_penalty=True,
                            layer_penalty=None,
                            flag_imbalance_penalty=False,
                            flag_importance_penalty=False,
                            penalty_aggregation="sum",
                            random_seed=100):
    size = f"{num_objs}_{num_vars}"

    sampling_type = f"npr{neg_pos_ratio}ms{min_samples}"
    sampling_data_path = path.resource / "xgb_data" / problem / size / split / sampling_type
    sampling_data_path.mkdir(parents=True, exist_ok=True)
    features_exists = sampling_data_path.joinpath(f"{pid}.npy").exists()

    labels_data_path = path.resource / "xgb_data" / problem / size / split / "labels" / label_type
    labels_data_path.mkdir(parents=True, exist_ok=True)
    labels_exists = labels_data_path.joinpath(f"{pid}.npy").exists()

    weights_type = ""
    if flag_layer_penalty:
        weights_type += f"{layer_penalty}-"
    weights_type += "1-" if flag_imbalance_penalty else "0-"
    weights_type += "1-" if flag_importance_penalty else "0-"
    weights_type += penalty_aggregation
    weights_data_path = path.resource / "xgb_data" / problem / size / split / sampling_type / weights_type
    weights_data_path.mkdir(parents=True, exist_ok=True)
    weights_exists = weights_data_path.joinpath(f"{pid}.npy").exists()

    print(f"Processed {pid}, Features - {features_exists}, Weights - {weights_exists}, Labels - {labels_exists}")

    def convert_bdd_to_xgb_data_knapsack():
        rng = random.Random(random_seed)
        features_lst = None if features_exists else []
        weights_lst = None if weights_exists else []
        labels_lst = None if labels_exists else []
        layer_weight_lst = get_layer_weights(flag_layer_penalty, layer_penalty, num_vars)

        inst_data, order, features, inst_features, var_features, num_var_features = None, None, None, None, None, None
        if not features_exists:
            # Read instance
            inst_data = get_instance_data(problem, size, split, pid)
            order = get_static_order(problem, order_type, inst_data)
            # Extract instance and variable features
            featurizer = get_featurizer(problem, FeaturizerConfig(norm_const=state_norm_const,
                                                                  raw=False,
                                                                  context=True))
            features = featurizer.get(inst_data)
            # Instance features
            inst_features = features["inst"][0]
            # Variable features. Reordered features based on ordering
            var_features = features["var"][order]
            num_var_features = features["var"].shape[1]

        # features_lst, labels_lst, weights_lst = [], [], []
        for lidx, layer in enumerate(bdd):
            # _features_lst, _labels_lst, _weights_lst = [], [], []
            pos_ids = [node_id for node_id, node in enumerate(layer) if node["pareto"] == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))

            # Subsample negative samples
            num_pos_samples = len(pos_ids)
            if neg_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = np.max([int(neg_pos_ratio * num_pos_samples),
                                          min_samples])
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            # Imbalance weights for the current layer
            pos_imb_wt = num_neg_samples / (num_pos_samples + num_neg_samples)
            neg_imb_wt = 1 - pos_imb_wt

            _parent_var_feat, _var_feat = None, None
            if not features_exists:
                # Variable features: Parent and current layer
                _parent_var_feat = -1 * np.ones(num_var_features) \
                    if lidx == 0 \
                    else var_features[lidx - 1]
                _var_feat = var_features[lidx]

            prev_layer = bdd[lidx - 1] if lidx > 0 else None
            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for i, node_id in enumerate(node_ids):
                node = layer[node_id]

                if not features_exists:
                    _node_feat, _parent_node_feat = extract_node_features("knapsack",
                                                                          lidx,
                                                                          node,
                                                                          prev_layer,
                                                                          inst_data,
                                                                          layer_norm_const=layer_norm_const,
                                                                          state_norm_const=state_norm_const)
                    features_lst.append(np.concatenate((inst_features,
                                                        _parent_var_feat,
                                                        _parent_node_feat,
                                                        _var_feat,
                                                        _node_feat)))

                if not labels_exists:
                    labels_lst.append(node["l"])

                if not weights_exists:
                    weights_lst.append(get_aggregated_weight(
                        aggregation=penalty_aggregation,
                        flag_layer_penalty=flag_layer_penalty,
                        layer_weight=layer_weight_lst[lidx],
                        flag_imbalance_penalty=flag_imbalance_penalty,
                        imb_wt=pos_imb_wt if i < num_pos_samples else neg_imb_wt,
                        flag_importance_penalty=flag_importance_penalty if i < num_pos_samples else False,
                        score=node["score"]))

        return features_lst, labels_lst, weights_lst

    if problem == "knapsack":
        data = convert_bdd_to_xgb_data_knapsack()
    else:
        raise ValueError("Invalid problem type!")

    features_lst, labels_lst, weights_lst = data
    if features_lst is not None:
        features_np = np.array(features_lst)
        np.save(open(sampling_data_path.joinpath(f"{pid}.npy"), "wb"), features_np)

    if labels_lst is not None:
        labels_np = np.array(labels_lst)
        np.save(open(labels_data_path.joinpath(f"{pid}.npy"), "wb"), labels_np)

    if weights_lst is not None:
        weights_np = np.array(weights_lst)
        np.save(open(weights_data_path.joinpath(f"{pid}.npy"), "wb"), weights_np)


def convert_bdd_to_xgb_mixed_data(problem,
                                  counter=None,
                                  bdd=None,
                                  num_objs=None,
                                  num_vars=None,
                                  split=None,
                                  pid=None,
                                  order_type=None,
                                  state_norm_const=1000,
                                  layer_norm_const=100,
                                  task="classification",
                                  label_type="binary",
                                  neg_pos_ratio=1,
                                  min_samples=0,
                                  flag_layer_penalty=True,
                                  layer_penalty=None,
                                  flag_imbalance_penalty=False,
                                  flag_importance_penalty=False,
                                  penalty_aggregation="sum",
                                  random_seed=100):
    size = f"{num_objs}_{num_vars}"
    dataset_type = "mixed"

    sampling_type = f"npr{neg_pos_ratio}ms{min_samples}"
    sampling_data_path = path.resource / "xgb_data" / problem / dataset_type / split / sampling_type
    sampling_data_path.mkdir(parents=True, exist_ok=True)
    features_exists = sampling_data_path.joinpath(f"{counter}.npy").exists()

    labels_data_path = path.resource / "xgb_data" / problem / dataset_type / split / "labels" / label_type
    labels_data_path.mkdir(parents=True, exist_ok=True)
    labels_exists = labels_data_path.joinpath(f"{counter}.npy").exists()

    weights_type = ""
    if flag_layer_penalty:
        weights_type += f"{layer_penalty}-"
    weights_type += "1-" if flag_imbalance_penalty else "0-"
    weights_type += "1-" if flag_importance_penalty else "0-"
    weights_type += penalty_aggregation
    weights_data_path = path.resource / "xgb_data" / problem / dataset_type / split / sampling_type / weights_type
    weights_data_path.mkdir(parents=True, exist_ok=True)
    weights_exists = weights_data_path.joinpath(f"{counter}.npy").exists()

    print(f"Processed {counter}, Features - {features_exists}, Weights - {weights_exists}, Labels - {labels_exists}")

    def convert_bdd_to_xgb_mixed_data_knapsack():
        rng = random.Random(random_seed)
        features_lst = None if features_exists else []
        weights_lst = None if weights_exists else []
        labels_lst = None if labels_exists else []
        layer_weight_lst = get_layer_weights(flag_layer_penalty, layer_penalty, num_vars)

        inst_data, order, features, inst_features, var_features, num_var_features = None, None, None, None, None, None
        if not features_exists:
            # Read instance
            inst_data = get_instance_data(problem, size, split, pid)
            order = get_static_order(problem, order_type, inst_data)
            # Extract instance and variable features
            featurizer = get_featurizer(problem, FeaturizerConfig(norm_const=state_norm_const,
                                                                  raw=False,
                                                                  context=True))
            features = featurizer.get(inst_data)
            # Instance features
            inst_features = features["inst"][0]
            # Variable features. Reordered features based on ordering
            var_features = features["var"][order]
            num_var_features = features["var"].shape[1]

        # features_lst, labels_lst, weights_lst = [], [], []
        for lidx, layer in enumerate(bdd):
            # _features_lst, _labels_lst, _weights_lst = [], [], []
            pos_ids = [node_id for node_id, node in enumerate(layer) if node["pareto"] == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))

            # Subsample negative samples
            num_pos_samples = len(pos_ids)
            if neg_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = np.max([int(neg_pos_ratio * num_pos_samples),
                                          min_samples])
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            # Imbalance weights for the current layer
            pos_imb_wt = num_neg_samples / (num_pos_samples + num_neg_samples)
            neg_imb_wt = 1 - pos_imb_wt

            _parent_var_feat, _var_feat = None, None
            if not features_exists:
                # Variable features: Parent and current layer
                _parent_var_feat = -1 * np.ones(num_var_features) \
                    if lidx == 0 \
                    else var_features[lidx - 1]
                _var_feat = var_features[lidx]

            prev_layer = bdd[lidx - 1] if lidx > 0 else None
            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for i, node_id in enumerate(node_ids):
                node = layer[node_id]

                if not features_exists:
                    _node_feat, _parent_node_feat = extract_node_features("knapsack",
                                                                          lidx,
                                                                          node,
                                                                          prev_layer,
                                                                          inst_data,
                                                                          layer_norm_const=layer_norm_const,
                                                                          state_norm_const=state_norm_const)
                    features_lst.append(np.concatenate((inst_features,
                                                        _parent_var_feat,
                                                        _parent_node_feat,
                                                        _var_feat,
                                                        _node_feat)))

                if not labels_exists:
                    labels_lst.append(node["l"])

                if not weights_exists:
                    weights_lst.append(get_aggregated_weight(
                        aggregation=penalty_aggregation,
                        flag_layer_penalty=flag_layer_penalty,
                        layer_weight=layer_weight_lst[lidx],
                        flag_imbalance_penalty=flag_imbalance_penalty,
                        imb_wt=pos_imb_wt if i < num_pos_samples else neg_imb_wt,
                        flag_importance_penalty=flag_importance_penalty if i < num_pos_samples else False,
                        score=node["score"]))

        return features_lst, labels_lst, weights_lst

    if problem == "knapsack":
        data = convert_bdd_to_xgb_mixed_data_knapsack()
    else:
        raise ValueError("Invalid problem type!")

    features_lst, labels_lst, weights_lst = data
    if features_lst is not None:
        features_np = np.array(features_lst)
        np.save(open(sampling_data_path.joinpath(f"{counter}.npy"), "wb"), features_np)

    if labels_lst is not None:
        labels_np = np.array(labels_lst)
        np.save(open(labels_data_path.joinpath(f"{counter}.npy"), "wb"), labels_np)

    if weights_lst is not None:
        weights_np = np.array(weights_lst)
        np.save(open(weights_data_path.joinpath(f"{counter}.npy"), "wb"), weights_np)


def get_nn_dataset(problem, size, split, pid, sampling_type, labels_type, weights_type, device):
    def get_dataset_knapsack():
        zf = zipfile.Path(path.resource / f"tensors/{problem}/{size}/{split}/{sampling_type}.zip")
        if zf.joinpath(f"{sampling_type}/n{pid}.pt").exists():
            return KnapsackBDDDataset(size=size,
                                      split=split,
                                      pid=pid,
                                      sampling_type=sampling_type,
                                      labels_type=labels_type,
                                      weights_type=weights_type,
                                      device=device)
        else:
            return None

    if problem == "knapsack":
        dataset = get_dataset_knapsack()
    else:
        raise ValueError("Invalid problem name!")

    return dataset


def get_xgb_dataset(problem, size, split, pid, neg_pos_ratio, min_samples):
    def get_dataset_knapsack():
        dtype = f"npr{neg_pos_ratio}ms{min_samples}"
        zf = zipfile.Path(path.resource / f"xgb_data/knapsack/{size}/{split}.zip")
        np_file = zf.joinpath(f"{split}/{dtype}/{pid}.npy")
        if np_file.exists():
            zf = zipfile.ZipFile(path.resource / f"xgb_data/knapsack/{size}/{split}.zip")
            with zf.open(f"{split}/{dtype}/{pid}.npy", "r") as fp:
                data = io.BytesIO(fp.read())
                np_array = np.load(data)
                return np_array
        else:
            return None

    dataset = None
    if problem == "knapsack":
        dataset = get_dataset_knapsack()

    return dataset


def get_dataloader(dataset, batch_size, shuffle=True):
    DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def statscore(preds=None, labels=None, threshold=0.5, round_upto=1, is_type="torch"):
    assert preds.shape == labels.shape
    if is_type == "numpy":
        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)

    # print(preds, labels)
    # Binarize preds
    preds = torch.round(preds, decimals=round_upto)
    one_idxs = preds >= threshold
    preds[one_idxs], preds[~one_idxs] = 1, 0

    # Binarize labels
    one_idxs = labels >= threshold
    labels[one_idxs], labels[~one_idxs] = 1, 0

    # Scores
    tp = ((labels == preds) & (labels == 1)).reshape(-1, 1)
    fn = ((labels != preds) & (labels == 1)).reshape(-1, 1)
    fp = ((labels != preds) & (labels == 0)).reshape(-1, 1)
    tn = ((labels == preds) & (labels == 0)).reshape(-1, 1)
    result = torch.cat((tp, fp, tn, fn), axis=1)
    result[result is True] = 1
    result[result is False] = 0
    if is_type == "numpy":
        result = result.numpy()

    return result


def update_scores(num_vars, scores_df, scores, tp, fp, tn, fn):
    tp += scores[:, 1].sum()
    fp += scores[:, 2].sum()
    tn += scores[:, 3].sum()
    fn += scores[:, 4].sum()
    for layer in range(num_vars):
        _scores = scores[scores[:, 0] == layer]
        if _scores.shape[0] > 0:
            _sum = _scores.sum(axis=0)
            scores_df.loc[layer, "TP"] += _sum[1]
            scores_df.loc[layer, "FP"] += _sum[2]
            scores_df.loc[layer, "TN"] += _sum[3]
            scores_df.loc[layer, "FN"] += _sum[4]
            # scores_df.loc[layer, "Support"] += _sum[5]

    return scores_df, tp, fp, tn, fn


def calculate_accuracy(tp, fp, tn, fn):
    correct = tp + tn
    total = tp + fp + tn + fn
    return correct / total, correct, total


def print_result(epoch,
                 split,
                 pid=None,
                 acc=None,
                 correct=None,
                 total=None,
                 inst_per_step=None,
                 is_best=None,
                 pre_space='\t'):
    is_best_str = " -- BEST ACC" if is_best else ""
    print(f"{pre_space}------------------------------------------------")
    if pid is not None and inst_per_step is not None:
        print(f"{pre_space}Inst: {pid}-{pid + inst_per_step}, "
              f"Acc: {acc:.2f}, Correct: {correct}, Total: {total}")
    else:
        print(f"{pre_space}Acc: {acc:.2f}, Correct: {correct}, Total: {total} {is_best_str}")


def get_log_dir_name(name,
                     size,
                     flag_layer_penalty,
                     layer_penalty,
                     flag_label_penalty,
                     label_penalty,
                     neg_pos_ratio,
                     order,
                     layer_norm_const,
                     state_norm_const):
    checkpoint_str = f"{name}-{size}/"

    if flag_layer_penalty:
        checkpoint_str += f"{layer_penalty}-"
    if flag_label_penalty:
        checkpoint_str += f"{int(label_penalty * 10)}-"

    checkpoint_str += f"{neg_pos_ratio}-"
    if neg_pos_ratio < 0:
        checkpoint_str += f"n{int(-1 * neg_pos_ratio)}-"
    else:
        checkpoint_str += f"{neg_pos_ratio}-"

    checkpoint_str += f"{order}-{layer_norm_const}-{state_norm_const}/"

    return checkpoint_str


def checkpoint(cfg, split, epoch=None, model=None, scores_df=None, is_best=None):
    mdl_path = path.resource / f"pretrained/nn/{cfg.prob.name}/{cfg.prob.size}"
    mdl_path.mkdir(parents=True, exist_ok=True)

    mdl_name = get_nn_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    hex = h.hexdigest()
    split_path = mdl_path.joinpath(f"{hex}/{split}")
    split_path.mkdir(parents=True, exist_ok=True)

    if model is not None:
        torch.save(model.state_dict(),
                   split_path.joinpath(f"model_{epoch}.ckpt"))

        if is_best:
            torch.save(model.state_dict(),
                       split_path.joinpath(f"model_best.ckpt"))

    if scores_df is not None:
        scores_df.to_csv(split_path.joinpath(f"scores_{epoch}.csv"), index=False)
        if is_best:
            scores_df.to_csv(split_path.joinpath(f"scores_best_{epoch}.csv"), index=False)

        # Normalize scores
        # scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
        #                          scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
        # scores_df["TP"] /= scores_df["Support"]
        # scores_df["FP"] /= scores_df["NSupport"]
        # scores_df["TN"] /= scores_df["NSupport"]
        # scores_df["FN"] /= scores_df["Support"]

        # score_name = f"scores_norm_{epoch}.csv"
        # scores_df_name = checkpoint_dir / score_name
        # scores_df.to_csv(scores_df_name, index=False)
        # if is_best:
        #     scores_df_name = checkpoint_dir / "scores_norm_best.csv"
        #     scores_df.to_csv(scores_df_name, index=False)


def checkpoint_test(cfg, scores_df):
    checkpoint_dir = path.resource / "experiments/"
    checkpoint_str = get_log_dir_name(cfg.prob.name,
                                      cfg.prob.size,
                                      cfg.train.flag_layer_penalty,
                                      cfg.train.layer_penalty,
                                      cfg.train.flag_label_penalty,
                                      cfg.train.label_penalty,
                                      cfg.train.neg_pos_ratio,
                                      cfg.prob.order,
                                      cfg.prob.layer_norm_const,
                                      cfg.prob.state_norm_const)
    checkpoint_str += f"{cfg.test.log_dir}"
    checkpoint_dir /= checkpoint_str
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    scores_df_name = checkpoint_dir / "scores.csv"
    scores_df.to_csv(scores_df_name, index=False)

    # Normalize scores
    scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
                             scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
    scores_df["TP"] /= scores_df["Support"]
    scores_df["FP"] /= scores_df["NSupport"]
    scores_df["TN"] /= scores_df["NSupport"]
    scores_df["FN"] /= scores_df["Support"]
    score_name = f"scores_norm.csv"
    scores_df_name = checkpoint_dir / score_name
    scores_df.to_csv(scores_df_name, index=False)


def handle_timeout(sig, frame):
    raise TimeoutError('Timeout')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_type):
    if device_type == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Training on ", device)

    return device


def setup_ddp(dist_backend="nccl", init_method="tcp://localhost:1234"):
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
    init_process_group(backend=dist_backend,
                       init_method=init_method,
                       world_size=world_size,
                       rank=global_rank)
    print("Process group ready!")
    print("From Rank: {}, ==> Making model...".format(global_rank))
    print()

    return world_size, global_rank, device_id


def get_device(distributed=False, init_method=None, dist_backend=None):
    device_str, pin_memory, master, device_id, world_size = "cpu", False, True, 0, 1
    if distributed:
        world_size, rank, device_id = setup_ddp(dist_backend=dist_backend, init_method=init_method)
        device_str = f"cuda:{device_id}"
        pin_memory = True
        master = rank == 0
    elif torch.cuda.is_available():
        device_str = "cuda"
        pin_memory = True
    device = torch.device(device_str)

    return device, device_str, pin_memory, master, device_id, world_size


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}_{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"


def get_split_datasets(pids, problem, size, split, sampling_type, labels_type, weights_type, device, dataset_dict=None):
    datasets = []
    # if dataset_dict is not None:
    #     for pid in pids:
    # if pid not in dataset_dict:
    #     dataset_dict[pid] = get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device)
    # if dataset_dict[pid] is not None:
    #     # print("Reading dataset ", pid)
    #     datasets.append(dataset_dict[pid])
    # datasets.append(get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device))
    # else:
    for pid in pids:
        dataset = get_nn_dataset(problem, size, split, pid, sampling_type, labels_type, weights_type, device)
        if dataset is not None:
            datasets.append(dataset)

    return datasets, dataset_dict


def label_bdd(bdd, labeling_scheme):
    for l in range(len(bdd)):
        for n in bdd[l]:
            if labeling_scheme == "binary":
                n["l"] = 1 if n["pareto"] else 0
            elif labeling_scheme == "mo":
                # Margin one
                n["l"] = 1 if n["pareto"] else -1
            elif labeling_scheme == "mos":
                # Margin one score
                n["l"] = 1 + n["score"] if n["pareto"] else -1
            elif labeling_scheme == "nms":
                # Negative margin score
                n["l"] = n["score"] if n["pareto"] else -1
            else:
                raise ValueError("Invalid labeling scheme!")

    return bdd


def get_xgb_model_name(max_depth=None,
                       eta=None,
                       min_child_weight=None,
                       subsample=None,
                       colsample_bytree=None,
                       objective=None,
                       num_round=None,
                       early_stopping_rounds=None,
                       evals=None,
                       eval_metric=None,
                       seed=None,
                       prob_name=None,
                       num_objs=None,
                       num_vars=None,
                       order=None,
                       layer_norm_const=None,
                       state_norm_const=None,
                       train_from_pid=None,
                       train_to_pid=None,
                       train_neg_pos_ratio=None,
                       train_min_samples=None,
                       train_flag_layer_penalty=None,
                       train_layer_penalty=None,
                       train_flag_imbalance_penalty=None,
                       train_flag_importance_penalty=None,
                       train_penalty_aggregation=None,
                       val_from_pid=None,
                       val_to_pid=None,
                       val_neg_pos_ratio=None,
                       val_min_samples=None,
                       val_flag_layer_penalty=None,
                       val_layer_penalty=None,
                       val_flag_imbalance_penalty=None,
                       val_flag_importance_penalty=None,
                       val_penalty_aggregation=None,
                       device=None):
    def get_model_name_knapsack():
        name = ""
        if max_depth is not None:
            name += f"{max_depth}-"
        if eta is not None:
            name += f"{eta}-"
        if min_child_weight is not None:
            name += f"{min_child_weight}-"
        if subsample is not None:
            name += f"{subsample}-"
        if colsample_bytree is not None:
            name += f"{colsample_bytree}-"
        if objective is not None:
            name += f"{objective}-"
        if num_round is not None:
            name += f"{num_round}-"
        if early_stopping_rounds is not None:
            name += f"{early_stopping_rounds}-"
        if type(evals) is list and len(evals):
            for eval in evals:
                name += f"{eval}"
        if type(eval_metric) is list and len(eval_metric):
            for em in eval_metric:
                name += f"{em}-"
        if seed is not None:
            name += f"{seed}"

        if prob_name is not None:
            name += f"{prob_name}-"
        if num_objs is not None:
            name += f"{num_objs}-"
        if num_vars is not None:
            name += f"{num_vars}-"
        if order is not None:
            name += f"{order}-"
        if layer_norm_const is not None:
            name += f"{layer_norm_const}-"
        if state_norm_const is not None:
            name += f"{state_norm_const}-"

        if train_from_pid is not None:
            name += f"{train_from_pid}-"
        if train_to_pid is not None:
            name += f"{train_to_pid}-"
        if train_neg_pos_ratio is not None:
            name += f"{train_neg_pos_ratio}-"
        if train_min_samples is not None:
            name += f"{train_min_samples}-"
        if train_flag_layer_penalty is not None:
            name += f"{train_flag_layer_penalty}-"
        if train_layer_penalty is not None:
            name += f"{train_layer_penalty}-"
        if train_flag_imbalance_penalty is not None:
            name += f"{train_flag_imbalance_penalty}-"
        if train_flag_importance_penalty is not None:
            name += f"{train_flag_importance_penalty}-"
        if train_penalty_aggregation is not None:
            name += f"{train_penalty_aggregation}-"

        if val_from_pid is not None:
            name += f"{val_from_pid}-"
        if val_to_pid is not None:
            name += f"{val_to_pid}-"
        if val_neg_pos_ratio is not None:
            name += f"{val_neg_pos_ratio}-"
        if val_min_samples is not None:
            name += f"{val_min_samples}-"
        if val_flag_layer_penalty is not None:
            name += f"{val_flag_layer_penalty}-"
        if val_layer_penalty is not None:
            name += f"{val_layer_penalty}-"
        if val_flag_imbalance_penalty is not None:
            name += f"{val_flag_imbalance_penalty}-"
        if val_flag_importance_penalty is not None:
            name += f"{val_flag_importance_penalty}-"
        if val_penalty_aggregation is not None:
            name += f"{val_penalty_aggregation}-"
        if device is not None:
            name += f"{device}"

        return name

    if prob_name == "knapsack":
        return get_model_name_knapsack()
    else:
        raise ValueError("Invalid problem!")


def get_nn_model_name(cfg):
    def get_nn_model_name_knapsack():
        name = ""
        name += str(cfg.pred_task)
        name += str(cfg.device)
        name += str(cfg.threshold)
        name += str(cfg.round_upto)
        name += str(cfg.prob.name)
        name += str(cfg.prob.size)
        name += str(cfg.prob.order)
        name += str(cfg.prob.layer_norm_const)
        name += str(cfg.prob.state_norm_const)
        name += str(cfg.train.epochs)
        name += str(cfg.train.from_pid)
        name += str(cfg.train.to_pid)
        name += str(cfg.train.inst_per_step)
        name += str(cfg.train.neg_pos_ratio)
        name += str(cfg.train.min_samples)
        name += str(cfg.train.batch_size)
        name += str(cfg.train.flag_layer_penalty)
        name += str(cfg.train.layer_penalty)
        name += str(cfg.train.flag_imbalance_penalty)
        name += str(cfg.train.flag_importance_penalty)
        name += str(cfg.train.penalty_aggregation)
        name += str(cfg.val.from_pid)
        name += str(cfg.val.to_pid)
        name += "-".join(map(str, cfg.mdl.ie.enc))
        name += "-".join(map(str, cfg.mdl.ie.agg))
        name += "-".join(map(str, cfg.mdl.ce.enc))
        name += "-".join(map(str, cfg.mdl.ce.agg))
        name += "-".join(map(str, cfg.mdl.pe.enc))
        name += "-".join(map(str, cfg.mdl.pe.agg))
        name += "-".join(map(str, cfg.mdl.ne))
        name += str(cfg.opt.name)
        name += str(cfg.opt.lr)
        name += str(cfg.loss.name)

        return name

    if cfg.prob.name == "knapsack":
        return get_nn_model_name_knapsack()
    else:
        raise ValueError("Invalid problem!")


def dict2cpu(stats):
    cpu_stats = {}
    for k, v in stats.items():
        if v is not None:
            cpu_stats[k] = v.cpu().numpy()
        else:
            cpu_stats[k] = None

    return cpu_stats


def reduce_epoch_time(epoch_time, device):
    epoch_time = torch.tensor(epoch_time, dtype=torch.float32, device=device)
    dist.all_reduce(epoch_time, dist.ReduceOp.AVG, async_op=False)
    return epoch_time
