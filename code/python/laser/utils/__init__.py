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
    def __init__(self, size=None, split=None, pid=None, dtype=None, device=None):
        super(KnapsackBDDDataset, self).__init__()

        zf = zipfile.ZipFile(resource_path / f"tensors/knapsack/{size}/{split}.zip")
        data = torch.load(zf.open(f"{split}/{dtype}/{pid}.pt"))
        self.node_feat = data["nf"].to(device)
        self.parent_feat = data["pf"].to(device)
        self.inst_feat = data["if"].to(device)
        self.wt_layer = data["wtlayer"].to(device)
        self.wt_label = data["wtlabel"].to(device)
        self.labels = data["label"].to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {'nf': self.node_feat[i],
                'pf': self.parent_feat[i],
                'if': self.inst_feat,
                'wtlayer': self.wt_layer[i],
                'wtlabel': self.wt_label[i],
                'label': self.labels[i]}


class MockConfig:
    norm_const = 1000
    raw = False
    context = True


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


def get_context_features(layer_idxs, inst_feat, num_objs, num_vars, device):
    max_lidx = np.max(layer_idxs)
    context = []
    for inst_idx, lidx in enumerate(layer_idxs):
        _inst_feat = inst_feat[inst_idx, :lidx, :]

        ranks = (torch.arange(lidx).reshape(-1, 1) + 1) / num_vars
        _context = torch.concat((_inst_feat, ranks.to(device)), axis=1)

        ranks_pad = torch.zeros(max_lidx - _inst_feat.shape[0], num_objs + 2).to(device)
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


def get_featurizer(problem, cfg):
    featurizer = None
    if problem == "knapsack":
        from laser.featurizer import KnapsackFeaturizer

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


def get_parent_features(problem, node, bdd, lidx, state_norm_const):
    def get_parent_features_knapsack():
        parents_feat = []

        for p in node['op']:
            parent_state = 0 if lidx == 0 else bdd[lidx - 1][p]['s'][0] / state_norm_const
            parents_feat.append([ONE_ARC, parent_state])

        for _ in node['zp']:
            # Parent state will be the same as the current state for zero-arc
            parent_state = node['s'][0] / state_norm_const
            parents_feat.append([ZERO_ARC, parent_state])

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
                               layer_penalty=None,
                               order_type=None,
                               state_norm_const=1000,
                               layer_norm_const=100,
                               neg_pos_ratio=1,
                               min_samples=0,
                               random_seed=100):
    size = f"{num_objs}_{num_vars}"

    def convert_bdd_to_tensor_dataset_knapsack():
        rng = random.Random(random_seed)
        # Load bdd
        _bdd = get_bdd_data(problem, size, split, pid) if bdd is None else bdd
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
        scores, labels = [], []
        for lidx, layer in enumerate(_bdd):
            _node_feat, _parents_feat = [], []
            _wt_layer, _wt_label = [], []
            _scores, _labels = [], []
            num_pos = np.sum([1 for node in layer if node['l'] == 1])
            for nidx, node in enumerate(layer):
                # Append to global
                if node['l'] == 1:
                    # Extract node feature
                    node_feat.append([node['s'][0] / state_norm_const,
                                      (lidx + 1) / layer_norm_const])
                    # Extract parent feature
                    parents_feat.append(get_parent_features(problem,
                                                            node,
                                                            _bdd,
                                                            lidx,
                                                            state_norm_const))
                    wt_layer.append(layer_weight[lidx])
                    wt_label.append(1)
                    scores.append(node["score"])
                    labels.append(1)
                    # Append to local
                else:
                    # Extract node feature
                    _node_feat.append([node['s'][0] / state_norm_const,
                                       (lidx + 1) / layer_norm_const])
                    # Extract parent feature
                    _parents_feat.append(get_parent_features(problem,
                                                             node,
                                                             _bdd,
                                                             lidx,
                                                             state_norm_const))
                    _wt_layer.append(layer_weight[lidx])
                    _wt_label.append(1 / neg_pos_ratio)
                    _scores.append(0)
                    _labels.append(0)

            # Select samples from local to append to the global
            if neg_pos_ratio == -1:
                # Select all negative samples
                node_feat.extend(_node_feat)
                parents_feat.extend(_parents_feat)
                wt_layer.extend(_wt_layer)
                wt_label.extend(_wt_label)
                scores.extend(_scores)
                labels.extend(_labels)
            else:
                neg_idxs = list(range(len(_labels)))
                rng.shuffle(neg_idxs)
                num_neg_samples = np.max([neg_pos_ratio * num_pos,
                                          min_samples])
                num_neg_samples = np.min([num_neg_samples,
                                          len(_labels)])
                for nidx in neg_idxs[:num_neg_samples]:
                    node_feat.append(_node_feat[nidx])
                    parents_feat.append(_parents_feat[nidx])
                    wt_layer.append(_wt_layer[nidx])
                    wt_label.append(_wt_label[nidx])
                    scores.append(_scores[nidx])
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
                    "label": np.array(labels).reshape(-1, 1),
                    "score": np.array(scores).reshape(-1, 1)}
        bdd_data = {k: np2tensor(v) for k, v in bdd_data.items()}

        return bdd_data

    data = None
    if problem == "knapsack":
        data = convert_bdd_to_tensor_dataset_knapsack()

    assert data is not None
    sampling_type = f"npr{neg_pos_ratio}ms{min_samples}"
    file_path = resource_path / "tensors" / problem / size / split / sampling_type
    file_path.mkdir(parents=True, exist_ok=True)
    file_path /= f"{pid}.pt"
    torch.save(data, file_path)


def convert_bdd_to_xgb_data(problem,
                            bdd=None,
                            num_objs=None,
                            num_vars=None,
                            split=None,
                            pid=None,
                            layer_penalty=None,
                            order_type=None,
                            state_norm_const=1000,
                            layer_norm_const=100,
                            neg_pos_ratio=1,
                            min_samples=0,
                            random_seed=100):
    size = f"{num_objs}_{num_vars}"

    def convert_bdd_to_xgb_data_knapsack():
        rng = random.Random(random_seed)
        layer_weight = get_layer_weights(layer_penalty, num_vars)

        # Read instance
        inst_data = get_instance_data(problem, size, split, pid)
        order = get_order(problem, order_type, inst_data)
        # Extract instance and variable features
        featurizer_conf = MockConfig()
        featurizer = get_featurizer(problem, featurizer_conf)
        features = featurizer.get(inst_data)
        # Instance features
        inst_features = features["inst"][0]
        # Variable features. Reordered features based on ordering
        features["var"] = features["var"][order]
        num_var_features = features["var"].shape[1]

        features_lst, labels_lst, weights_lst = [], [], []
        for lidx, layer in enumerate(bdd):
            _features_lst, _labels_lst, _weights_lst = [], [], []
            num_pos = np.sum([1 for node in layer if node['l'] == 1])

            # Parent variable features
            _parent_var_feat = -1 * np.ones(num_var_features) \
                if lidx == 0 \
                else features["var"][lidx - 1]

            for node in layer:
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
                        prev_layer = lidx - 1
                        prev_node_idx = node["op"][0]
                        prev_state = bdd[prev_layer][prev_node_idx]["s"][0]
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

                if node["l"] > 0:
                    # Features
                    features_lst.append(np.concatenate((inst_features,
                                                        _parent_var_feat,
                                                        _parent_node_feat,
                                                        features["var"][lidx],
                                                        _node_feat)))
                    labels_lst.append(node["l"])
                    weights_lst.append(layer_weight[lidx])
                else:
                    # Features
                    _features_lst.append(np.concatenate((inst_features,
                                                         _parent_var_feat,
                                                         _parent_node_feat,
                                                         features["var"][lidx],
                                                         _node_feat)))
                    _labels_lst.append(node["l"])
                    _weights_lst.append(layer_weight[lidx])

            # Select samples from local to append to the global
            if neg_pos_ratio == -1:
                # Select all negative samples
                features_lst.extend(_features_lst)
                labels_lst.extend(_labels_lst)
                weights_lst.extend(_weights_lst)
            else:
                neg_idxs = list(range(len(_labels_lst)))
                rng.shuffle(neg_idxs)
                num_neg_samples = np.max([int(neg_pos_ratio * num_pos),
                                          min_samples])
                num_neg_samples = np.min([num_neg_samples,
                                          len(_labels_lst)])
                for nidx in neg_idxs[:num_neg_samples]:
                    features_lst.append(_features_lst[nidx])
                    labels_lst.append(_labels_lst[nidx])
                    weights_lst.append(_weights_lst[nidx])

        return features_lst, labels_lst, weights_lst

    data = None
    if problem == "knapsack":
        data = convert_bdd_to_xgb_data_knapsack()

    assert data is not None
    # sampling_type = f"npr{neg_pos_ratio}ms{min_samples}"
    # file_path = resource_path / "xgb_data" / problem / size / split / sampling_type
    # file_path.mkdir(parents=True, exist_ok=True)
    # file_path /= f"{pid}.pt"
    # torch.save(data, file_path)
    data = (np.array(data[0]), np.array(data[1]).reshape(-1, 1), np.array(data[2]).reshape(-1, 1))
    data = np.hstack(data)

    return data


def get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device):
    def get_dataset_knapsack():
        dtype = f"npr{neg_pos_ratio}ms{min_samples}"
        zf = zipfile.Path(resource_path / f"tensors/knapsack/{size}/{split}.zip")
        if zf.joinpath(f"{split}/{dtype}/{pid}.pt").exists():
            return KnapsackBDDDataset(size=size,
                                      split=split,
                                      pid=pid,
                                      dtype=dtype,
                                      device=device)
        else:
            return None

    dataset = None
    if problem == "knapsack":
        dataset = get_dataset_knapsack()

    return dataset


def get_dataloader(dataset, batch_size, shuffle=True):
    DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def update_scores(scores_df, scores, tp, fp, tn, fn):
    tp += scores[:, 1].sum()
    fp += scores[:, 2].sum()
    tn += scores[:, 3].sum()
    fn += scores[:, 4].sum()
    for layer in range(40):
        _scores = scores[scores[:, 0] == layer]
        if _scores.shape[0] > 0:
            _sum = _scores.sum(axis=0)
            scores_df.loc[layer, "TP"] += _sum[1]
            scores_df.loc[layer, "FP"] += _sum[2]
            scores_df.loc[layer, "TN"] += _sum[3]
            scores_df.loc[layer, "FN"] += _sum[4]
            scores_df.loc[layer, "Support"] += _sum[5]

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
    checkpoint_dir = resource_path / "experiments/"
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
    random.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_device(device_type):
    if device_type == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Training on ", device)

    return device


def get_split_datasets(pids, problem, size, split, neg_pos_ratio, min_samples, device, dataset_dict=None):
    datasets = []
    if dataset_dict is not None:
        for pid in pids:
            if pid not in dataset_dict:
                dataset_dict[pid] = get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device)
            if dataset_dict[pid] is not None:
                # print("Reading dataset ", pid)
                datasets.append(dataset_dict[pid])
            datasets.append(get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device))
    else:
        for pid in pids:
            dataset = get_dataset(problem, size, split, pid, neg_pos_ratio, min_samples, device)
            if dataset is not None:
                datasets.append(dataset)

    return datasets, dataset_dict
