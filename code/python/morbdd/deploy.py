import hashlib
import json
import multiprocessing as mp
import time

import hydra
import libbddenvv1
# import networkx as nx
import numpy as np
import pandas as pd
# import torch
# from torchmetrics.classification import BinaryStatScores

from morbdd import resource_path
from morbdd.utils import get_instance_data
from morbdd.utils import get_order
from morbdd.utils import get_xgb_model_name
from morbdd.utils import label_bdd
from morbdd.utils import statscore
import gurobipy as gp
from morbdd.heuristics import stitch


def call_get_model_name(cfg):
    if cfg.deploy.mdl == "xgb":
        mdl_cfg = cfg[cfg.deploy.mdl]

        return get_xgb_model_name(max_depth=mdl_cfg.max_depth,
                                  eta=mdl_cfg.eta,
                                  min_child_weight=mdl_cfg.min_child_weight,
                                  subsample=mdl_cfg.subsample,
                                  colsample_bytree=mdl_cfg.colsample_bytree,
                                  objective=mdl_cfg.objective,
                                  num_round=mdl_cfg.num_round,
                                  early_stopping_rounds=mdl_cfg.early_stopping_rounds,
                                  evals=mdl_cfg.evals,
                                  eval_metric=mdl_cfg.eval_metric,
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
        for node in layer:
            if np.round(node["pred"], round_upto) >= threshold:
                is_connected = True
                node["conn"] = True
            else:
                node["conn"] = False
    else:
        # Check if we have a high scoring node. If yes, then check if at least one of the parents is also high scoring.
        for node in layer:
            is_node_connected = False
            node["conn"] = False
            if np.round(node["pred"], round_upto) >= threshold:
                for op in node["op"]:
                    if prev_layer[op]["conn"]:
                        is_connected = True
                        is_node_connected = True
                        node["conn"] = True
                        break

                if not is_node_connected:
                    for zp in node["zp"]:
                        if prev_layer[zp]["conn"]:
                            is_connected = True
                            node["conn"] = True
                            break

    return is_connected


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


def get_prediction_stats(bdd, pred_stats_per_layer, threshold=0.5, round_upto=1):
    for lidx, layer in enumerate(bdd):
        labels = np.array([node["l"] for node in layer])
        assert np.max(labels) >= threshold

        preds = np.array([node["pred"] for node in layer])
        score = statscore(preds=preds, labels=labels, threshold=threshold, round_upto=round_upto,
                          is_type="numpy")
        tp, fp, tn, fn = np.sum(score[:, 0]), np.sum(score[:, 1]), np.sum(score[:, 2]), np.sum(score[:, 3])
        pred_stats_per_layer[lidx + 1][1] += tp
        pred_stats_per_layer[lidx + 1][2] += fp
        pred_stats_per_layer[lidx + 1][3] += tn
        pred_stats_per_layer[lidx + 1][4] += fn

    return pred_stats_per_layer


def save_stats_per_layer(cfg, pred_stats_per_layer, mdl_hex):
    df = pd.DataFrame(pred_stats_per_layer,
                      columns=["layer", "tp", "fp", "tn", "fn"])
    name = resource_path / f"predictions/{cfg.deploy.mdl}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    # if cfg.deploy.stitching_heuristic == "min_resistance":
    #     name /= f"{cfg.deploy.select_all_upto}-mrh{cfg.deploy.lookahead}-spl.csv"
    # elif cfg.deploy.stitching_heuristic:
    #     name /= f"{cfg.deploy.select_all_upto}-sph-spl.csv"
    # else:
    #     raise ValueError("Invalid heuristic!")
    name /= "spl.csv"
    df.to_csv(name, index=False)


def save_bdd_data(cfg, pids, bdd_data, mdl_hex):
    out_path = resource_path / f"predictions/{cfg.deploy.mdl}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    if cfg.deploy.stitching_heuristic == "min_resistance":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-mrh{cfg.deploy.lookahead}"
    elif cfg.deploy.stitching_heuristic == "shortest_path":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-sph"
    elif cfg.deploy.stitching_heuristic == "mip":
        disconnected_prefix = f"{cfg.deploy.select_all_upto}-mip"
    else:
        raise ValueError("Invalid heuristic!")

    bdd_stats = []
    bdd_stats_disconnected = []
    for pid, data in zip(pids, bdd_data):
        time_stitching, time_mip, count_stitching, was_disconnected, inc, rnc, iac, rac, num_comparisons, sol, _time = data

        sol_pred_path = out_path / f"{disconnected_prefix}-sols_pred" \
            if was_disconnected else out_path / "sols_pred"
        sol_pred_path.mkdir(exist_ok=True, parents=True)
        sol_path = sol_pred_path / f"sol_{pid}.json"
        with open(sol_path, "w") as fp:
            json.dump(sol, fp)
        time_path = sol_pred_path / f"time_{pid}.json"
        with open(time_path, "w") as fp:
            json.dump(_time, fp)

        if was_disconnected:
            bdd_stats_disconnected.append([cfg.prob.size,
                                           pid,
                                           cfg.deploy.split,
                                           1,
                                           count_stitching,
                                           time_stitching,
                                           time_mip,
                                           cfg.deploy.stitching_heuristic,
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
        else:
            bdd_stats.append([cfg.prob.size,
                              pid,
                              cfg.deploy.split,
                              0,
                              0,
                              0,
                              0,
                              "",
                              "",
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

    columns = ["size",
               "pid",
               "split",
               "was_disconnected",
               "count_stitching",
               "time_stitching",
               "time_mip",
               "stitching_heuristic",
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
               "pred_pareto"]
    if len(bdd_stats):
        df = pd.DataFrame(bdd_stats, columns=columns)
        df.to_csv(out_path / f"pred_result_{pids[0]}.csv", index=False)

    if len(bdd_stats_disconnected):
        df = pd.DataFrame(bdd_stats_disconnected, columns=columns)
        df.to_csv(out_path / f"{disconnected_prefix}-pred_result_{pids[0]}.csv", index=False)


def compute_pareto_frontier_on_pareto_bdd(cfg, env, pareto_states, inst_data, order):
    if cfg.prob.name == "knapsack":
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
    else:
        raise ValueError("Invalid problem name!")

    env.compute_pareto_frontier_with_pruning(pareto_states)

    return env


def worker(rank, cfg, mdl_hex):
    env = libbddenvv1.BDDEnv()

    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    pids, bdd_data = [], []
    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        print(pid)
        # Read instance
        inst_data = get_instance_data(cfg.prob.name, cfg.prob.size, cfg.deploy.split, pid)
        order = get_order(cfg.prob.name, cfg.deploy.order_type, inst_data)

        # Load BDD
        bdd_path = resource_path / (f"predictions/{cfg.deploy.mdl}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/"
                                    f"{mdl_hex}/pred_bdd/{pid}.json")
        print(bdd_path)
        if not bdd_path.exists():
            continue
        bdd = json.load(open(bdd_path, "r"))
        bdd = label_bdd(bdd, cfg.deploy.label)
        pred_stats_per_layer = get_prediction_stats(bdd,
                                                    pred_stats_per_layer,
                                                    threshold=cfg.deploy.threshold,
                                                    round_upto=cfg.deploy.round_upto)

        # Check connectedness of predicted Pareto BDD and perform stitching if necessary
        was_disconnected, total_time_stitching, time_mip, count_stitching = False, 0, 0, 0
        for lidx, layer in enumerate(bdd):
            prev_layer = bdd[lidx - 1] if lidx > 0 else None
            is_connected = check_connectedness(prev_layer,
                                               layer,
                                               threshold=cfg.deploy.threshold,
                                               round_upto=cfg.deploy.round_upto)
            if not is_connected:
                print(f"Disconnected {pid}, layer: ", lidx)
                was_disconnected = True
                count_stitching += 1
                bdd, total_time_stitching, time_mip = stitch("knapsack",
                                                             cfg,
                                                             bdd,
                                                             lidx,
                                                             total_time_stitching)

        # cfg.deploy.stitching_heuristic = "mip"
        # bdd, total_time_stitching, time_mip = stitch("knapsack", cfg, bdd, lidx, total_time_stitching)

        if ((was_disconnected is False and cfg.deploy.process_connected) or
                (was_disconnected is True and cfg.deploy.process_disconnected)):
            # Compute Pareto frontier on predicted Pareto BDD
            pareto_states = get_pareto_states_per_layer(bdd,
                                                        threshold=cfg.deploy.threshold,
                                                        round_upto=cfg.deploy.round_upto)
            env = compute_pareto_frontier_on_pareto_bdd(cfg, env, pareto_states, inst_data, order)

            # Extract run info
            _data = get_run_data_from_env(env, cfg.deploy.order_type, was_disconnected)
            _data1 = [total_time_stitching, time_mip, count_stitching]
            _data1.extend(_data)

            pids.append(pid)
            bdd_data.append(_data1)
            print(f'Processed: {pid}, was_disconnected: {_data[0]}, n_sols: {len(_data[-2]["x"])}')

    return pids, bdd_data, pred_stats_per_layer


@hydra.main(version_base="1.2", config_path="./configs", config_name="deploy.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()
    print(f"Using model: ", mdl_hex)
    # Deploy model
    # pool = mp.Pool(processes=cfg.deploy.num_processes)
    # results = []
    # for rank in range(cfg.deploy.num_processes):
    #     results.append(pool.apply_async(worker, args=(rank, cfg, mdl_hex)))

    results = [worker(0, cfg, mdl_hex)]

    # Fetch results
    results = [r.get() for r in results]

    pids, bdd_data = [], []
    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    for r in results:
        pids.extend(r[0])
        bdd_data.extend(r[1])
        pred_stats_per_layer[:, 1:] += r[2][:, 1:]

    if len(pids):
        # Save results
        save_bdd_data(cfg, pids, bdd_data, mdl_hex)
        # save_stats_per_layer(cfg, pred_stats_per_layer, mdl_hex)


if __name__ == '__main__':
    main()
