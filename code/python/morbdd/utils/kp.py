from operator import itemgetter

import numpy as np

from morbdd import ResourcePaths as path
from morbdd.utils import FeaturizerConfig, get_featurizer
from morbdd.utils import read_from_zip
from morbdd.featurizer import KnapsackFeaturizer


class LayerToFeatureConverter:
    def __init__(self, with_parent=False):
        self.featurizer = KnapsackFeaturizer(FeaturizerConfig())
        self.with_parent = with_parent
        self.inst_data = None
        self.order = None
        self.features = None
        self.inst_features = None
        self.var_features = None
        self.num_var_features = 0
        self.layer_norm_const = 100
        self.state_norm_const = 1000

    def reset(self, inst_data, order):
        self.inst_data = inst_data
        self.order = order

        self.features = self.featurizer.get(inst_data)
        # Instance features
        self.inst_features = self.features["inst"][0]
        # Variable features. Reordered features based on ordering
        self.var_features = self.features["var"][order]
        self.num_var_features = self.var_features.shape[1]

    def convert(self, lidx, layer):
        """lidx starts from 0 for layer 1"""
        features_lst = []

        # Parent variable features
        var_feat = self.var_features[lidx]
        for node in layer:
            node_feat = get_bdd_node_features_gbt_rank(
                lidx,
                node,
                self.inst_data["capacity"],
                self.layer_norm_const,
                self.state_norm_const,
            )

            features_lst.append(
                np.concatenate((self.inst_features, var_feat, node_feat))
            )

        return np.array(features_lst)


def get_instance_path(seed, n_objs, n_vars, split, pid, name="knapsack", prefix="kp"):
    return (
        path.inst
        / f"{name}/{n_objs}_{n_vars}/{split}/{prefix}_{seed}_{n_objs}_{n_vars}_{pid}.dat"
    )


def read_instance(archive, inst):
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


def get_instance_data(name, prefix, seed, size, split, pid, suffix=".dat"):
    archive = path.inst / f"{name}/{size}.zip"
    inst = f"{size}/{split}/{prefix}_{seed}_{size}_{pid}{suffix}"
    data = read_instance(archive, inst)

    return data


def get_static_order(order_type, data):
    if order_type == "MinWt":
        idx_weight = [(i, w) for i, w in enumerate(data["weight"])]
        idx_weight.sort(key=itemgetter(1))

        return np.array([i[0] for i in idx_weight])
    elif order_type == "MaxRatio":
        min_profit = np.min(data["value"], 0)
        profit_by_weight = [v / w for v, w in zip(min_profit, data["weight"])]
        idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
        idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)

        return np.array([i[0] for i in idx_profit_by_weight])
    elif order_type == "Lex":
        return np.arange(data["n_vars"])


def get_bdd_node_features(
    lidx,
    node,
    prev_layer,
    capacity,
    layer_norm_const,
    state_norm_const,
    with_parent=False,
):
    # Node features
    norm_state = node["s"][0] / state_norm_const
    state_to_capacity = node["s"][0] / capacity
    layers_to_go = (layer_norm_const - lidx) / layer_norm_const
    node_feat = np.array([norm_state, state_to_capacity, layers_to_go])

    return node_feat


def get_bdd_node_features_gbt_rank(
    lidx, node, capacity, layer_norm_const, state_norm_const
):
    # Node features
    norm_state = node["s"][0] / state_norm_const
    state_to_capacity = node["s"][0] / capacity
    layers_to_go = (layer_norm_const - lidx) / layer_norm_const
    node_feat = np.array([norm_state, state_to_capacity, layers_to_go])

    return node_feat
