import json
import random
import zipfile
from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from laser import resource_path

ZERO_ARC = -1
ONE_ARC = 1


class KnapsackBDDDataset(Dataset):
    def __init__(self, size, split, pid):
        super(KnapsackBDDDataset, self).__init__()

        data = torch.load(resource_path / f"datasets/knapsack/{size}/{split}/{pid}.pt")
        self.node_feat = data["nf"]
        self.parent_feat = data["pf"]
        self.inst_feat = data["if"]
        self.wt_layer = data["wtlayer"]
        self.wt_label = data["wtlabel"]
        self.labels = data["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'nf': self.node_feat[i],
                'pf': self.parent_feat[i],
                'if': self.inst_feat,
                'wtlayer': self.wt_layer[i],
                'wtlabel': self.wt_label[i],
                'label': self.labels[i]}


def read_from_zip(archive, file, format="raw"):
    zf = zipfile.ZipFile(archive)
    raw_data = zf.open(file, "r")

    data = None
    if format == "raw":
        data = raw_data
    elif format == "json":
        data = json.load(raw_data)

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


def read_instance(problem, archive, inst):
    data = None
    if problem == "knapsack":
        data = read_instance_knapsack(archive, inst)

    return data


def get_instance_prefix(problem):
    prefix = None
    if problem == 'knapsack':
        prefix = 'kp_7'

    return prefix


def get_instance_data(problem, size, split, pid):
    prefix = get_instance_prefix(problem)
    archive = resource_path / f"instances/{problem}/{size}.zip"
    inst = f'{size}/{split}/{prefix}_{size}_{pid}.dat'
    data = read_instance(problem, archive, inst)

    return data


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


def get_layer_weights(penalty, num_vars):
    lidxs = [lidx / num_vars for lidx in range(num_vars)]

    # Set layer weights
    if penalty == "const":
        layer_weight = [1 for _ in num_vars]
    elif penalty == "linear":
        layer_weight = [1 - lidx for lidx in lidxs]
    elif penalty == "exponential":
        layer_weight = [np.exp(-0.5 * lidx) for lidx in lidxs]
    elif penalty == "linearE":
        layer_weight = [(np.exp(-0.5) - 1) * lidx + 1
                        for lidx in lidxs]
    elif penalty == "quadratic":
        layer_weight = [(np.exp(-0.5) - 1) * (lidx ** 2) + 1
                        for lidx in lidxs]
    elif penalty == "sigmoidal":
        layer_weight = [(1 + np.exp(-0.5)) / (1 + np.exp(lidx - 0.5))
                        for lidx in lidxs]
    else:
        raise ValueError("Invalid layer penalty scheme!")

    return layer_weight


def get_bdd_data(problem, size, split, pid):
    archive = resource_path / f"bdds/{problem}/{size}.zip"
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


def get_order(problem, order_type, data):
    order = None
    if problem == 'knapsack':
        order = get_knapsack_order(order_type, data)

    assert order is not None

    return order


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


def get_parent_features(problem, node, bdd, lidx, state_norm_const):
    def get_parent_features_knapsack():
        parents_feat = []

        for p in node['op']:
            parent_state = 0 if lidx == 0 else bdd[lidx - 1][p]['s'][0] / state_norm_const
            parents_feat.append([ONE_ARC, parent_state])

        for p in node['zp']:
            # Parent state will be the same as the current state for zero-arc
            parent_state = node['s'][0] / state_norm_const
            parents_feat.append([ZERO_ARC, parent_state])

        return parents_feat

    if problem == "knapsack":
        get_parent_features_knapsack()


def np2tensor(data):
    return torch.from_numpy(data).float()


def convert_bdd_to_tensor_data(problem,
                               num_objs=None,
                               num_vars=None,
                               split=None,
                               pid=None,
                               layer_penalty=None,
                               order_type=None,
                               state_norm_const=1000,
                               layer_norm_const=100,
                               neg_pos_ratio=1,
                               random_seed=100):
    size = f"{num_objs}_{num_vars}"

    def convert_bdd_to_tensor_dataset_knapsack():
        rng = random.Random(random_seed)
        # Load bdd
        bdd = get_bdd_data(problem, size, split, pid)
        layer_weight = get_layer_weights(layer_penalty, num_vars)

        # Get instance features
        data = get_instance_data(problem, size, split, pid)
        order = get_order(problem, order_type, data)
        inst_feat = get_instance_features(problem,
                                          data,
                                          state_norm_const=state_norm_const)
        # Reset order of the instance
        inst_feat = inst_feat[:, order]

        # Get node and parent features
        node_feat, parents_feat = [], []
        wt_layer, wt_label = [], []
        labels = []
        for lidx, layer in enumerate(bdd):
            _node_feat, _parents_feat = [], []
            _wt_layer, _wt_label = [], []
            _labels = []
            num_pos = np.sum([1 for node in layer if node['l'] == 1])
            for nidx, node in enumerate(layer):
                # Append to global
                if node['l'] == 1:
                    # Extract node feature
                    node_feat.append([node['s'][0] / state_norm_const,
                                      (lidx + 1) / layer_norm_const])
                    # Extract parent feature
                    parents_feat.append(get_parent_features(node,
                                                            bdd,
                                                            lidx,
                                                            state_norm_const))
                    wt_layer.append(layer_weight[lidx])
                    wt_label.append(1)
                    labels.append(1)
                    # Append to local
                else:
                    # Extract node feature
                    _node_feat.append([node['s'][0] / state_norm_const,
                                       (lidx + 1) / layer_norm_const])
                    # Extract parent feature
                    _parents_feat.append(get_parent_features(problem,
                                                             node,
                                                             bdd,
                                                             lidx,
                                                             state_norm_const))
                    _wt_layer.append(layer_weight[lidx])
                    _wt_label.append(1 / neg_pos_ratio)
                    _labels.append(0)

            # Select samples from local to append to the global
            if neg_pos_ratio == -1:
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

        bdd_data = {"nf": np.array(node_feat),
                    "pf": np.array(parents_feat_padded),
                    "if": np.array(inst_feat).T,
                    "wtlayer": np.array(wt_layer),
                    "wtlabel": np.array(wt_label),
                    "label": np.array(labels).reshape(-1, 1)}
        bdd_data = {k: np2tensor(v) for k, v in bdd_data.items()}

        return bdd_data

    data = None
    if problem == "knapsack":
        data = convert_bdd_to_tensor_dataset_knapsack()

    assert data is not None
    file_path = resource_path / "datasets" / problem / size / split
    file_path.mkdir(parents=True, exist_ok=True)
    file_path /= f"{pid}.pt"
    torch.save(data, file_path)


def get_dataset(problem, split, pid):
    def get_dataset_knapsack():
        return KnapsackBDDDataset(split=split, pid=pid)

    if problem == "knapsack":
        get_dataset_knapsack()


def get_dataloader(dataset, batch_size, shuffle=True):
    DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def update_scores(scores_df, scores, tp, fp, tn, fn):
    tp += scores[:, 1].sum()
    fp += scores[:, 2].sum()
    tn += scores[:, 3].sum()
    fn += scores[:, 4].sum()
    for i in range(scores.shape[0]):
        layer = int(scores[i][0])
        scores_df.loc[layer, "TP"] += scores[i][1]
        scores_df.loc[layer, "FP"] += scores[i][2]
        scores_df.loc[layer, "TN"] += scores[i][3]
        scores_df.loc[layer, "FN"] += scores[i][4]
        scores_df.loc[layer, "Support"] += scores[i][5]

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
                 is_best=None):
    is_best_str = " -- BEST ACC" if is_best else ""
    if pid is not None and inst_per_step is not None:
        print(f"\tEpoch: {epoch}, Inst: {pid}-{pid + inst_per_step}, "
              f"{split} acc: {acc:.2f}, {correct}, {total}")
    else:
        print(f"\tEpoch: {epoch}, {split} acc: {acc:.2f}, {correct}, {total} {is_best_str}")


def get_log_dir_name(cfg):
    checkpoint_str = f"{cfg.prob.name}-{cfg.prob.size}/"

    if cfg.train.flag_layer_penalty:
        checkpoint_str += f"{cfg.train.layer_penalty}-"
    if cfg.train.flag_label_penalty:
        checkpoint_str += f"{int(cfg.train.label_penalty * 10)}-"

    checkpoint_str += f"{cfg.train.neg_pos_ratio}-"
    if cfg.val.neg_pos_ratio < 0:
        checkpoint_str += f"n{int(-1 * cfg.val.neg_pos_ratio)}-"
    else:
        checkpoint_str += f"{cfg.val.neg_pos_ratio}-"

    checkpoint_str += f"{cfg.prob.order}-{cfg.prob.layer_norm_const}-{cfg.prob.state_norm_const}/"

    return checkpoint_str


def checkpoint(cfg, split, epoch=None, model=None, scores_df=None, is_best=None):
    checkpoint_dir = resource_path / "experiments/"
    checkpoint_str = get_log_dir_name(cfg)
    checkpoint_str += f"{cfg[split].log_dir}"
    checkpoint_dir /= checkpoint_str
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if model is not None:
        model_name = f"model_{epoch}.ckpt"
        model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), model_path)

        if is_best:
            model_name = f"model_best.ckpt"
            model_path = checkpoint_dir / model_name
            torch.save(model.state_dict(), model_path)

    if scores_df is not None:
        score_name = f"scores_{epoch}.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)
        if is_best:
            scores_df_name = checkpoint_dir / "scores_best.csv"
            scores_df.to_csv(scores_df_name, index=False)

        # Normalize scores
        scores_df["NSupport"] = (scores_df["TP"] + scores_df["FP"] +
                                 scores_df["TN"] + scores_df["FN"]) - scores_df["Support"]
        scores_df["TP"] /= scores_df["Support"]
        scores_df["FP"] /= scores_df["NSupport"]
        scores_df["TN"] /= scores_df["NSupport"]
        scores_df["FN"] /= scores_df["Support"]

        score_name = f"scores_norm_{epoch}.csv"
        scores_df_name = checkpoint_dir / score_name
        scores_df.to_csv(scores_df_name, index=False)
        if is_best:
            scores_df_name = checkpoint_dir / "scores_norm_best.csv"
            scores_df.to_csv(scores_df_name, index=False)


def checkpoint_test(cfg, scores_df):
    checkpoint_dir = resource_path / "experiments/"
    checkpoint_str = get_log_dir_name(cfg)
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
