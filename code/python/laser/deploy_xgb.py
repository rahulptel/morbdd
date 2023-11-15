import hashlib
import json
import multiprocessing as mp

import hydra
import libbddenvv1
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import BinaryStatScores

from laser import resource_path
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import get_xgb_model_name
from laser.utils import label_bdd


def call_get_model_name(cfg):
    return get_xgb_model_name(max_depth=cfg.max_depth,
                              eta=cfg.eta,
                              objective=cfg.objective,
                              num_round=cfg.num_round,
                              early_stopping_rounds=cfg.early_stopping_rounds,
                              evals=cfg.evals,
                              eval_metric=cfg.eval_metric,
                              seed=cfg.seed,
                              prob_name=cfg.prob.name,
                              num_objs=cfg.prob.num_objs,
                              num_vars=cfg.prob.num_vars,
                              order=cfg.prob.order,
                              layer_norm_const=cfg.prob.layer_norm_const,
                              state_norm_const=cfg.prob.state_norm_const,
                              train_from_pid=cfg.train.from_pid,
                              train_to_pid=cfg.train.to_pid,
                              train_neg_pos_ratio=cfg.train.neg_pos_ratio,
                              train_min_samples=cfg.train.min_samples,
                              train_flag_layer_penalty=cfg.train.flag_layer_penalty,
                              train_layer_penalty=cfg.train.layer_penalty,
                              train_flag_imbalance_penalty=cfg.train.flag_imbalance_penalty,
                              train_flag_importance_penalty=cfg.train.flag_importance_penalty,
                              train_penalty_aggregation=cfg.train.penalty_aggregation,
                              val_from_pid=cfg.val.from_pid,
                              val_to_pid=cfg.val.to_pid,
                              val_neg_pos_ratio=cfg.val.neg_pos_ratio,
                              val_min_samples=cfg.val.min_samples,
                              val_flag_layer_penalty=cfg.val.flag_layer_penalty,
                              val_layer_penalty=cfg.val.layer_penalty,
                              val_flag_imbalance_penalty=cfg.val.flag_imbalance_penalty,
                              val_flag_importance_penalty=cfg.val.flag_importance_penalty,
                              val_penalty_aggregation=cfg.val.penalty_aggregation,
                              device=cfg.device)


def check_connectedness(prev_layer, layer, threshold=0.5, round_upto=1):
    is_connected = False
    if prev_layer is None:
        # On the first layer, we only check if there exists at least one node with a score higher than threshold
        # to check for connectedness as the root node is always selected.
        scores = [np.round(node["pred"], round_upto) for node in layer]
        is_connected = np.max(scores) >= threshold
    else:
        # Check if we have a high scoring node. If yes, then check if at least one of the parents is also high scoring.
        for node in layer:
            if np.round(node["pred"], round_upto) >= threshold:
                for prev_one_id in node["op"]:
                    if np.round(prev_layer[prev_one_id]["pred"], round_upto) >= threshold:
                        is_connected = True
                        break
                for prev_zero_id in node["zp"]:
                    if np.round(prev_layer[prev_zero_id]["pred"], round_upto) >= threshold:
                        is_connected = True
                        break

    return is_connected


def extend_paths(layer, partial_paths):
    new_partial_paths = []
    for node_idx, node in enumerate(layer):
        for parent_idx in node["op"]:
            for path in partial_paths:
                if path[-1] == parent_idx:
                    new_path = path[:]
                    new_path.append(node_idx)
                    new_partial_paths.append(new_path)

        for parent_idx in node["zp"]:
            for path in partial_paths:
                if path[-1] == parent_idx:
                    new_path = path[:]
                    new_path.append(node_idx)
                    new_partial_paths.append(new_path)

    return new_partial_paths


# def calculate_path_resistance(paths, cl, nl, threshold=0.5, round_upto=1):
def calculate_path_resistance(path, layers, threshold=0.5, round_upto=1):
    resistance = 0
    for node_idx, layer in zip(path[1:], layers):
        pred_score = layer[node_idx]["pred"]
        _resistance = 0 if np.round(pred_score, round_upto) >= threshold else threshold - pred_score
        resistance += _resistance

    return resistance


def stitch_layer(bdd, lidx, select_all_upto, heuristic, lookahead, threshold=0.5, round_upto=1):
    pl = bdd[lidx - 1] if lidx > 0 else None
    layers = [bdd[i] for i in range(lidx, lidx + lookahead)]

    flag_stitched_layer = False
    # If BDD is disconnected on the first layer, select both nodes.
    if lidx < select_all_upto:
        for node in layers[0]:
            if np.round(node["pred"], round_upto) < threshold:
                node["prev_pred"] = float(node["pred"])
                node["pred"] = float(threshold + 0.001)
                flag_stitched_layer = True
    else:
        if heuristic == "min_resistance":
            partial_paths = [[node_idx] for node_idx, node in enumerate(pl) if
                             np.round(node["pred"], round_upto) >= threshold]

            for i in range(lookahead - 1):
                partial_paths = extend_paths(layers[i], partial_paths)
            paths = extend_paths(layers[-1], partial_paths)

            resistances = [calculate_path_resistance(path, layers, threshold=threshold, round_upto=round_upto)
                           for path in paths]

            resistances, paths = zip(*sorted(zip(resistances, paths), key=lambda x: x[0]))
            k = 1
            for r in resistances[1:]:
                if r > resistances[0]:
                    break
                else:
                    k += 1

            for path in paths[:k]:
                for node_idx, layer in zip(path[1:], layers):
                    if np.round(layer[node_idx]["pred"], round_upto) < threshold:
                        layer[node_idx]["prev_pred"] = float(layer[node_idx]["pred"])
                        layer[node_idx]["pred"] = float(threshold + 0.001)
                        flag_stitched_layer = True
        else:
            raise ValueError("Invalid heuristic!")

    return flag_stitched_layer, bdd


def get_pareto_states_per_layer(bdd, threshold=0.5, round_upto=1):
    pareto_states_per_layer = []
    for layer in bdd:
        _pareto_states = []
        for node in layer:
            if np.round(node["pred"], round_upto) >= threshold:
                _pareto_states.append(node["s"][0])

        pareto_states_per_layer.append(_pareto_states)

    return pareto_states_per_layer


def get_run_data_from_env(env, order_type, was_disconnected):
    sol = {"x": env.x_sol,
           "z": env.z_sol,
           "ot": order_type}

    data = [was_disconnected,
            env.initial_node_count,
            env.reduced_node_count,
            env.initial_arcs_count,
            env.reduced_arcs_count,
            env.num_comparisons,
            sol,
            env.time_result]

    return data


def get_prediction_stats(bdd, scorer, pred_stats_per_layer, threshold=0.5, round_upto=1):
    for lidx, layer in enumerate(bdd):
        labels = np.array([np.round(node["l"], round_upto) >= threshold for node in layer])
        preds = np.array([np.round(node["pred"], round_upto) >= threshold for node in layer])
        tp, fp, tn, fn, sup = scorer(torch.from_numpy(preds),
                                     torch.from_numpy(labels))

        pred_stats_per_layer[lidx + 1][1] += tp
        pred_stats_per_layer[lidx + 1][2] += fp
        pred_stats_per_layer[lidx + 1][3] += tn
        pred_stats_per_layer[lidx + 1][4] += fn

    return pred_stats_per_layer


def save_bdd(problem, size, split, pid, bdd, mdl_hex):
    pred_bdd_path = resource_path / f"predictions/xgb/{problem}/{size}/{split}/{mdl_hex}/pred_bdd"
    pred_bdd_path.mkdir(parents=True, exist_ok=True)
    pred_bdd_path = pred_bdd_path / f"{pid}.json"
    with open(pred_bdd_path, "w") as fp:
        json.dump(bdd, fp)


def save_stats_per_layer(cfg, pred_stats_per_layer, mdl_hex):
    df = pd.DataFrame(pred_stats_per_layer,
                      columns=["layer", "tp", "fp", "tn", "fn"])
    name = resource_path / f"predictions/xgb/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    name /= f"{cfg.deploy.select_all_upto}-{cfg.deploy.lookahead}-spl.csv"
    df.to_csv(name, index=False)


def save_bdd_data(cfg, pids, bdd_data, mdl_hex):
    out_path = resource_path / f"predictions/xgb/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    sol_pred_path = out_path / f"{cfg.deploy.select_all_upto}-{cfg.deploy.lookahead}-sols_pred"
    sol_pred_path.mkdir(exist_ok=True, parents=True)

    bdd_stats = []
    for pid, data in zip(pids, bdd_data):
        was_connected, inc, rnc, iac, rac, num_comparisons, sol, _time = data

        sol_path = sol_pred_path / f"sol_{pid}.json"
        with open(sol_path, "w") as fp:
            json.dump(sol, fp)

        time_path = sol_pred_path / f"time_{pid}.json"
        with open(time_path, "w") as fp:
            json.dump(_time, fp)

        bdd_stats.append([cfg.prob.size,
                          pid,
                          cfg.deploy.split,
                          int(was_connected),
                          cfg.deploy.lookahead,
                          cfg.deploy.select_all_upto,
                          len(sol["x"]),
                          inc,
                          rnc,
                          iac,
                          rac,
                          num_comparisons,
                          _time["compilation"],
                          _time["reduction"],
                          _time["pareto"]])

    df = pd.DataFrame(bdd_stats, columns=["size",
                                          "pid",
                                          "split",
                                          "was_disconnected",
                                          "lookahead",
                                          "select_all_upto",
                                          "pred_nnds",
                                          "pred_inc",
                                          "pred_rnc",
                                          "pred_iac",
                                          "pred_rac",
                                          "pred_num_comparisons",
                                          "pred_compile",
                                          "pred_reduce",
                                          "pred_pareto"])
    df.to_csv(out_path / f"{cfg.deploy.select_all_upto}-{cfg.deploy.lookahead}-pred_result.csv", index=False)


def worker(rank, cfg, mdl_hex):
    env = libbddenvv1.BDDEnv()
    scorer = BinaryStatScores()

    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    pids, bdd_data = [], []
    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        # Read instance
        inst_data = get_instance_data(cfg.prob.name, cfg.prob.size, cfg.deploy.split, pid)
        order = get_order(cfg.prob.name, cfg.deploy.order_type, inst_data)

        # Load BDD
        bdd_path = resource_path / f"predictions/xgb/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}/pred_bdd/{pid}.json"
        bdd = json.load(open(bdd_path, "r"))
        bdd = label_bdd(bdd, cfg.deploy.label)

        # Check connectedness of predicted Pareto BDD and perform stitching if necessary
        was_disconnected = False
        for lidx, layer in enumerate(bdd):
            prev_layer = bdd[lidx - 1] if lidx > 0 else None
            is_connected = check_connectedness(prev_layer,
                                               layer,
                                               threshold=cfg.deploy.threshold,
                                               round_upto=cfg.deploy.round_upto)
            if not is_connected:
                # print("Disconnected, layer: ", lidx)
                was_disconnected = True
                flag_stitched_layer, bdd = stitch_layer(bdd,
                                                        lidx,
                                                        cfg.deploy.select_all_upto,
                                                        cfg.deploy.stitching_heuristic,
                                                        cfg.deploy.lookahead,
                                                        threshold=cfg.deploy.threshold,
                                                        round_upto=cfg.deploy.round_upto)
                if flag_stitched_layer is False:
                    break

        # Compute Pareto frontier on predicted Pareto BDD
        env.set_knapsack_inst(cfg.prob.num_vars,
                              cfg.prob.num_objs,
                              inst_data['value'],
                              inst_data['weight'],
                              inst_data['capacity'])
        env.initialize_run(cfg.bin.problem_type,
                           cfg.bin.preprocess,
                           cfg.bin.bdd_type,
                           cfg.bin.maxwidth,
                           order)
        pareto_states = get_pareto_states_per_layer(bdd,
                                                    threshold=cfg.deploy.threshold,
                                                    round_upto=cfg.deploy.round_upto)
        env.compute_pareto_frontier_with_pruning(pareto_states)

        # Extract run info
        pred_stats_per_layer = get_prediction_stats(bdd,
                                                    scorer,
                                                    pred_stats_per_layer,
                                                    threshold=cfg.deploy.threshold,
                                                    round_upto=cfg.deploy.round_upto)
        _data = get_run_data_from_env(env, cfg.deploy.order_type, was_disconnected)

        pids.append(pid)
        bdd_data.append(_data)
        print(f'Processed: {pid}, was_disconnected: {_data[0]}, n_sols: {len(_data[-2]["x"])}')

    return pids, bdd_data, pred_stats_per_layer


@hydra.main(version_base="1.2", config_path="./configs", config_name="deploy_xgb.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()

    # Deploy model
    pool = mp.Pool(processes=cfg.deploy.num_processes)
    results = []
    for rank in range(cfg.deploy.num_processes):
        results.append(pool.apply_async(worker, args=(rank, cfg, mdl_hex)))

    # results = worker(0, cfg, mdl_hex)

    # Fetch results
    results = [r.get() for r in results]
    pids, bdd_data = [], []
    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    for r in results:
        pids.extend(r[0])
        bdd_data.extend(r[1])
        pred_stats_per_layer[:, 1:] += r[2][:, 1:]

    # Save results
    save_bdd_data(cfg, pids, bdd_data, mdl_hex)
    save_stats_per_layer(cfg, pred_stats_per_layer, mdl_hex)


if __name__ == '__main__':
    main()
