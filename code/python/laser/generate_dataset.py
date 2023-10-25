import multiprocessing as mp

import hydra
import numpy as np
import xgboost as xgb

from laser import resource_path
from laser.utils import convert_bdd_to_tensor_data
from laser.utils import get_featurizer
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import read_from_zip
from laser.utils import get_layer_weights
import shutil


class MockConfig:
    norm_const = 1000
    raw = False
    context = True


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


def worker(rank, cfg):
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print(f"Processing pid {pid}...")
        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.label)

        print(f"\tRank {rank}: Converting BDD:{pid} to tensor dataset...")
        convert_bdd_to_tensor_data(cfg.prob,
                                   bdd=bdd,
                                   num_objs=cfg.num_objs,
                                   num_vars=cfg.num_vars,
                                   split=cfg.split,
                                   pid=pid,
                                   layer_penalty=cfg.layer_penalty,
                                   order_type=cfg.order_type,
                                   state_norm_const=cfg.state_norm_const,
                                   layer_norm_const=cfg.layer_norm_const,
                                   neg_pos_ratio=cfg.neg_pos_ratio,
                                   min_samples=cfg.min_samples,
                                   random_seed=cfg.seed)


def worker_xgb(rank, cfg):
    features_lst, labels_lst, weights_lst = [], [], []
    layer_weight = get_layer_weights(cfg.layer_penalty, cfg.num_vars)

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print("\tReading sol...")
        # print(f"Processing pid {pid}...")
        # Read instance
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_order(cfg.prob, cfg.order_type, data)

        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.label)

        # Extract instance and variable features
        featurizer_conf = MockConfig()
        featurizer = get_featurizer(cfg.prob, featurizer_conf)
        features = featurizer.get(data)
        # Instance features
        inst_features = features["inst"][0]
        # Variable features
        # Reorder features based on ordering
        features["var"] = features["var"][order]
        num_var_features = features["var"].shape[1]

        for lidx, layer in enumerate(bdd):
            # Parent variable features
            _parent_var_feat = -1 * np.ones(num_var_features) if lidx == 0 else features["var"][lidx - 1]

            for node in layer:
                norm_state = node["s"][0] / cfg.state_norm_const
                state_to_capacity = node["s"][0] / data["capacity"]
                _node_feat = np.array([norm_state, state_to_capacity, lidx])

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
                        _parent_node_feat.append(prev_state / cfg.state_norm_const)
                        _parent_node_feat.append(prev_state / data["capacity"])
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

                features_lst.append(np.concatenate((inst_features,
                                                    features["var"][lidx],
                                                    _parent_var_feat,
                                                    _parent_node_feat,
                                                    _node_feat)))
                labels_lst.append(node["l"])
                weights_lst.append(layer_weight[lidx])

        print(f"Processed {pid}...")
    return features_lst, labels_lst, weights_lst


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    pool = mp.Pool(processes=cfg.num_processes)
    results = []
    if cfg.dtype == "Tensor":
        for rank in range(cfg.num_processes):
            results.append(pool.apply_async(worker, args=(rank, cfg)))

        for r in results:
            r.get()

    elif cfg.dtype == "DMatrix":
        for rank in range(cfg.num_processes):
            results.append(pool.apply_async(worker_xgb, args=(rank, cfg)))

        # a, b, c = worker_xgb(0, cfg)
        results = [r.get() for r in results]
        features = np.concatenate([r[0] for r in results])
        label = np.concatenate([r[1] for r in results])
        weight = np.concatenate([r[2] for r in results])
        xgbd = xgb.DMatrix(features, label=label, weight=weight)
        name = f"{cfg.split}-{cfg.from_pid}-{cfg.to_pid}"
        if cfg.flag_layer_penalty:
            name += f"-{cfg.layer_penalty}"
        if cfg.flag_label_penalty:
            name += f"-{cfg.label_penalty}"
        name += ".buffer"
        xgbd.save_binary(name)

        file_path = resource_path / f"xgb_dataset/{cfg.prob}/{cfg.size}"
        file_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(name, file_path)


if __name__ == '__main__':
    main()
