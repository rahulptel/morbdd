from operator import itemgetter

import numpy as np

from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip


def get_instance_path(seed, n_objs, n_vars, split, pid, name="knapsack", prefix="kp"):
    return path.inst / f'{name}/{n_objs}_{n_vars}/{split}/{prefix}_{seed}_{n_objs}_{n_vars}_{pid}.dat'


def read_instance(archive, inst):
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


def get_instance_data(name, prefix, seed, size, split, pid, suffix=".dat"):
    archive = path.inst / f"{name}/{size}.zip"
    inst = f'{size}/{split}/{prefix}_{seed}_{size}_{pid}{suffix}'
    data = read_instance(archive, inst)

    return data


def get_static_order(order_type, data):
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


def get_bdd_node_features(lidx, node, prev_layer, capacity, layer_norm_const, state_norm_const, with_parent=False):
    # Node features
    norm_state = node["s"][0] / state_norm_const
    state_to_capacity = node["s"][0] / capacity
    layers_to_go = (layer_norm_const - lidx) / layer_norm_const
    node_feat = np.array([norm_state, state_to_capacity, layers_to_go])

    if with_parent:
        # Parent node features
        parent_node_feat = []
        if lidx == 0:
            parent_node_feat.extend([1, -1, -1, -1, -1, -1])
        else:
            # 1 implies parent of the one arc
            parent_node_feat.append(1)
            if len(node["op"]) > 0:
                prev_node_idx = node["op"][0]
                prev_state = prev_layer[prev_node_idx]["s"][0]
                parent_node_feat.append(prev_state / state_norm_const)
                parent_node_feat.append(prev_state / capacity)
            else:
                parent_node_feat.append(-1)
                parent_node_feat.append(-1)

            # -1 implies parent of the zero arc
            parent_node_feat.append(-1)
            if len(node["zp"]) > 0:
                parent_node_feat.append(norm_state)
                parent_node_feat.append(state_to_capacity)
            else:
                parent_node_feat.append(-1)
                parent_node_feat.append(-1)
        parent_node_feat = np.array(parent_node_feat)
        node_feat = np.concatenate([node_feat, parent_node_feat])

    return node_feat
