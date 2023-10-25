import json

import hydra
import libbddenvv1
import numpy as np

from laser import resource_path
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import read_from_zip
from laser.utils import convert_bdd_to_tensor_data
import multiprocessing as mp
from laser.utils import get_featurizer
from laser.utils import read_instance
from laser.utils import get_order
import xgboost as xgb


class MockConfig:
    norm_const = 1000
    raw = False
    context = True


def get_pareto_states_per_layer(weight, x):
    pareto_state_scores = []
    for i in range(1, x.shape[1]):
        x_partial = x[:, :i].reshape(-1, i)
        w_partial = weight[:i].reshape(i, 1)
        wt_dist = np.dot(x_partial, w_partial)
        pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
        pareto_score = pareto_score / pareto_score.sum()
        pareto_state_scores.append((pareto_state, pareto_score))

    return pareto_state_scores


def label_bdd(bdd, pareto_state_scores):
    assert len(pareto_state_scores) == len(bdd)

    for l in range(len(bdd)):
        pareto_states, pareto_scores = pareto_state_scores[l]

        for n in bdd[l]:
            index = np.where(pareto_states == n["s"])[0]
            if len(index):
                n["l"] = 1
                n["score"] = pareto_scores[index[0]]
            else:
                n["l"] = 0
                n["score"] = 0

    return bdd


def worker(rank, cfg):
    env = libbddenvv1.BDDEnv()

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print(f"Processing pid {pid}...")
        # Read instance
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_order(cfg.prob, cfg.order_type, data)

        # print("\tReading sol...")
        archive = resource_path / f"sols/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        sol = read_from_zip(archive, file, format="json")
        # Ignore instances not solved within time limit
        if sol is None:
            return

        # Extract BDD before reduction
        # print("\tExtracting non-reduced BDD...")
        env.set_knapsack_inst(cfg.num_vars,
                              cfg.num_objs,
                              data['value'],
                              data['weight'],
                              data['capacity'])
        bdd = env.get_bdd(cfg.problem_type, order)

        # Label BDD
        # print("\tLabelling BDD...")
        weight = np.array(data['weight'])[order]
        pareto_state_scores = get_pareto_states_per_layer(weight, np.array(sol["x"]))
        bdd = label_bdd(bdd, pareto_state_scores)

        # Save
        print("\tSaving BDD...")
        file_path = resource_path / f"bdds/{cfg.prob}/{cfg.size}/{cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(bdd, fp)

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
    env = libbddenvv1.BDDEnv()
    inp = []
    out = []

    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print(f"Processing pid {pid}...")
        # Read instance
        data = get_instance_data(cfg.prob, cfg.size, cfg.split, pid)
        order = get_order(cfg.prob, cfg.order_type, data)

        # print("\tReading sol...")
        archive = resource_path / f"sols/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        sol = read_from_zip(archive, file, format="json")
        # Ignore instances not solved within time limit
        if sol is None:
            return

        # Extract BDD before reduction
        # print("\tExtracting non-reduced BDD...")
        env.set_knapsack_inst(cfg.num_vars,
                              cfg.num_objs,
                              data['value'],
                              data['weight'],
                              data['capacity'])
        bdd = env.get_bdd(cfg.problem_type, order)

        # Label BDD
        # print("\tLabelling BDD...")
        weight = np.array(data['weight'])[order]
        capacity = data["capacity"]
        pareto_state_scores = get_pareto_states_per_layer(weight, np.array(sol["x"]))
        bdd = label_bdd(bdd, pareto_state_scores)

        feat_conf = MockConfig()
        featurizer = get_featurizer(cfg.prob, feat_conf)
        features = featurizer.get(data)

        for lidx, layer in enumerate(bdd):
            print("Layer", lidx)

            # Parent variable features
            if lidx == 0:
                _parent_var_feat = -1 * np.ones(8)
            else:
                _parent_var_feat = features["var"][lidx - 1]

            for node in layer:
                norm_state = node["s"][0] / capacity
                _node_feat = np.array([norm_state, lidx])

                _parent_node_feat = []
                if lidx == 0:
                    _parent_node_feat.extend([1, 0, -1, 0])
                else:
                    _parent_node_feat.append(1)
                    if len(node["op"]) > 1:
                        _parent_node_feat.append(bdd[lidx - 1][node["op"][0]]["s"][0] / capacity)
                    else:
                        _parent_node_feat.append(0)

                    _parent_node_feat.append(-1)
                    if len(node["zp"]) > 0:
                        _parent_node_feat.append(norm_state)
                    else:
                        _parent_node_feat.append(0)
                _parent_node_feat = np.array(_parent_node_feat)

                # print(features["inst"][0].shape,
                #       features["var"][lidx].shape,
                #       _parent_var_feat.shape,
                #       _parent_node_feat.shape,
                #       _node_feat.shape)

                inp.append(np.concatenate((features["inst"][0],
                                           features["var"][lidx],
                                           _parent_var_feat,
                                           _parent_node_feat,
                                           _node_feat)))
                out.append(node["l"])

            if lidx == 3:
                break


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

        results = [r.get() for r in results]
        features = np.concatenate([r[0] for r in results])
        label = np.concatenate([r[1] for r in results])
        weight = np.concatenate([r[2] for r in results])
        xgbd = xgb.DMatrix(features, label=label, weight=weight)
        xgbd.save_binary(f"{cfg.split}.buffer")


if __name__ == '__main__':
    main()
