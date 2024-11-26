import io
import json
import math
import os
import random
import zipfile

import numpy as np
import torch

from morbdd import ResourcePaths as path


# import pygmo as pg
class MetricCalculator:
    def __init__(self, n_objs, eps=0.1, delta=0.1):
        self.eps = eps
        self.delta = delta
        self.ref_point = None
        self.set_ref_point(n_objs)

    def set_ref_point(self, n_objs):
        self.ref_point = np.zeros(n_objs)

    @staticmethod
    def compute_cardinality(z, z_pred):
        z, z_pred = np.array(z), np.array(z_pred)
        assert z.shape[1] == z_pred.shape[1]

        if z.shape[0] == 0:
            print("True PF not available!")
            return {"card": -1, "precision": -1}

        if z_pred.shape[0] == 0:
            print("Predicted PF not available!")
            return {"card": 0, "precision": 0}

        # Defining a data type
        rows, cols = z.shape
        data_type_z = {
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [z.dtype],
        }

        rows, cols = z_pred.shape
        data_type_z_pred = {
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [z_pred.dtype],
        }

        # Finding intersection
        found_ndps = np.intersect1d(z.view(data_type_z), z_pred.view(data_type_z_pred))

        return {
            "cardinality": found_ndps.shape[0] / z.shape[0],
            "cardinality_raw": found_ndps.shape[0],
            "precision": found_ndps.shape[0] / z_pred.shape[0],
        }

    # def compute_approx_hv(self, seed, z_norm):
    #     hv_algo = pg.bf_fpras(eps=self.eps, delta=self.delta, seed=seed)
    #     hv = pg.hypervolume(z_norm)
    #     hv_approx = hv.compute(self.ref_point, hv_algo=hv_algo)
    #
    #     return {'hv_approx': hv_approx}


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


class FeaturizerConfig:
    def __init__(self, norm_const=1000, raw=False, context=True):
        self.norm_const = norm_const
        self.raw = raw
        self.context = context


class Result:
    def __init__(self):
        self.total_time = None
        self.orig_size = None
        self.restricted_size = None
        self.reduced_size = None
        self.orig_width = None
        self.reduced_width = None
        self.restricted_width = None
        self.cardinality = None
        self.cardinality_raw = None
        self.precision = None
        self.pred_pf = None
        self.pred_sol = None
        self.n_pred_pf = None
        self.build_time = None
        self.pareto_time = None


def zipdir(path, ziph):
    # Iterate over all the files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Create the relative path to maintain the folder structure
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


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
    file_path = (
        path.dataset / f"{cfg.prob.name}/{cfg.model.type}/{cfg.prob.size}/{cfg.split}"
    )
    prefix = get_dataset_prefix(cfg.with_parent, cfg.layer_weight, cfg.neg_to_pos_ratio)
    file_path /= prefix

    return file_path


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
    data = {"value": [], "n_vars": 0, "n_cons": 1, "n_objs": 3}
    data["weight"], data["capacity"] = [], 0

    raw_data = read_from_zip(archive, inst, format="raw")
    data["n_vars"] = int(raw_data.readline())
    data["n_objs"] = int(raw_data.readline())
    for _ in range(data["n_objs"]):
        data["value"].append(list(map(int, raw_data.readline().split())))
    data["weight"].extend(list(map(int, raw_data.readline().split())))
    data["capacity"] = int(raw_data.readline().split()[0])

    return data


def read_instance_indepset(archive, inst):
    if inst.split(".")[-1] == "npz":
        data = read_from_zip(archive, inst, format="npz")
    else:
        raw_data = read_from_zip(archive, inst)

        data = {"obj_coeffs": [], "cons_coeffs": [], "rhs": []}

        data["n_vars"], data["n_cons"] = list(
            map(int, raw_data.readline().strip().split())
        )
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
    if problem == "knapsack" or problem == "knapsackc":
        prefix = "kp_7"
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

    inst = f"{size}/{split}/{prefix}_{size}_{pid}.{suffix}"
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
    return [(np.exp(-0.5) - 1) * (lidx**2) + 1 for lidx in lidxs]


def get_layer_weights_sigmoidal(lidxs):
    return [(1 + np.exp(-0.5)) / (1 + np.exp(lidx - 0.5)) for lidx in lidxs]


def get_layer_weights(flag_penalty, penalty, num_vars):
    lidxs = [lidx / num_vars for lidx in range(num_vars)]
    get_layer_weights_fn = {
        "const": get_layer_weights_const,
        "linear": get_layer_weights_linear,
        "exponential": get_layer_weights_exponential,
        "linearE": get_layer_weights_linearE,
        "sigmoidal": get_layer_weights_sigmoidal,
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
        layer_weight.extend(layer_weight_b[: num_vars - upto_layer])

    return layer_weight


def get_bdd_data(problem, size, split, pid):
    archive = path.bdd / f"{problem}/{size}.zip"
    zf = zipfile.ZipFile(archive)
    fp = zf.open(f"{size}/{split}/{pid}.json", "r")
    bdd = json.load(fp)

    return bdd


def get_instance_features(problem, data, state_norm_const=None):
    def get_knapsack_instance_features():
        _feat = np.concatenate(
            (np.array(data["value"]), np.array(data["weight"]).reshape(1, -1)), axis=0
        )

        assert state_norm_const is not None
        _feat = _feat / state_norm_const

        return _feat

    feat = None
    if problem == "knapsack":
        feat = get_knapsack_instance_features()

    assert feat is not None
    return feat


def extract_node_features(
    problem,
    lidx,
    node,
    prev_layer,
    inst_data,
    layer_norm_const=None,
    state_norm_const=None,
):
    def extract_node_features_knapsack():
        # Node features
        norm_state = node["s"][0] / state_norm_const
        state_to_capacity = node["s"][0] / inst_data["capacity"]
        _node_feat = np.array(
            [norm_state, state_to_capacity, (lidx + 1) / layer_norm_const]
        )

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


def get_aggregated_weight(
    aggregation="sum",
    flag_layer_penalty=False,
    layer_weight=1,
    flag_imbalance_penalty=False,
    imb_wt=1,
    flag_importance_penalty=False,
    score=0,
):
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


def handle_timeout(sig, frame):
    raise TimeoutError("Timeout")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}_{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"


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


def compute_cardinality(true_pf=None, pred_pf=None):
    z, z_pred = np.array(true_pf), np.array(pred_pf)
    assert z.shape[1] == z_pred.shape[1]

    if z_pred.shape[0] == 0:
        return 0
    else:
        # Defining a data type
        rows, cols = z.shape
        dt_z = {
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [z.dtype],
        }

        rows, cols = z_pred.shape
        dt_z_pred = {
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [z_pred.dtype],
        }

        # Finding intersection
        found_ndps = np.intersect1d(z.view(dt_z), z_pred.view(dt_z_pred))

        return found_ndps.shape[0]


def compute_dd_size(dd):
    return np.sum([len(l) for l in dd])


def compute_dd_width(dd):
    return np.max([len(l) for l in dd])


def is_better(prev_best, new_result, metric):
    if (
        metric == "f1"
        or metric == "accuracy"
        or metric == "precision"
        or metric == "recall"
    ):
        if new_result > prev_best:
            return True

    elif metric == "loss":
        if new_result < prev_best:
            return True

    else:
        raise ValueError("Invalid metric!")

    return False


def initialize_eval_metric(metric):
    if (
        metric == "f1"
        or metric == "accuracy"
        or metric == "precision"
        or metric == "recall"
    ):
        return 0

    elif metric == "loss":
        return np.infty


def adjust_learning_rate(cfg, step, optimizer, warmup_steps, decay_steps):
    """Linearly increase learning rate and then decrease the learning rate using ReduceLROnPlateau scheduler."""
    lr = cfg.lr
    if cfg.warmup > 0 and step < warmup_steps:
        lr = cfg.lr * (step + 1) / warmup_steps
    elif cfg.decay_lr and step > decay_steps:
        lr = cfg.min_lr
    elif cfg.decay_lr and step <= decay_steps:
        decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        lr = cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
