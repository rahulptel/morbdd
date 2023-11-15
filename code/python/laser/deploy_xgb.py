import copy
import json
import multiprocessing as mp

import hydra
import libbddenvv1
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from laser import resource_path
from laser.utils import FeaturizerConfig
from laser.utils import get_featurizer
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import label_bdd
from laser.utils import read_from_zip
from laser.utils import get_xgb_model_name
import hashlib
from torchmetrics.classification import BinaryStatScores
from laser.utils import extract_node_features


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


def convert_bdd_to_xgb_data_deploy(problem,
                                   bdd=None,
                                   inst_data=None,
                                   order=None,
                                   state_norm_const=1000,
                                   layer_norm_const=100):
    def convert_bdd_to_xgb_data_deploy_knapsack():
        # Extract instance and variable features
        featurizer_conf = FeaturizerConfig()
        featurizer = get_featurizer(problem, featurizer_conf)
        features = featurizer.get(inst_data)
        # Instance features
        inst_features = features["inst"][0]
        # Variable features. Reordered features based on ordering
        var_features = features["var"][order]
        num_var_features = var_features.shape[1]

        features_lst, labels_lst = [], []
        for lidx, layer in enumerate(bdd):
            # Parent variable features
            _parent_var_feat = -1 * np.ones(num_var_features) \
                if lidx == 0 \
                else var_features[lidx - 1]
            _var_feat = var_features[lidx]

            prev_layer = bdd[lidx - 1]
            for node in layer:
                _node_feat, _parent_node_feat = extract_node_features("knapsack",
                                                                      lidx,
                                                                      node,
                                                                      prev_layer,
                                                                      inst_data,
                                                                      layer_norm_const=layer_norm_const,
                                                                      state_norm_const=state_norm_const)

                # Features
                features_lst.append(np.concatenate((inst_features,
                                                    _parent_var_feat,
                                                    _parent_node_feat,
                                                    _var_feat,
                                                    _node_feat)))
                labels_lst.append(node["l"])

        return features_lst, labels_lst

    data = None
    if problem == "knapsack":
        data = convert_bdd_to_xgb_data_deploy_knapsack()

    assert data is not None
    return np.array(data[0]), np.array(data[1])


def load_model(cfg, mdl_hex):
    mdl_path = resource_path / f"pretrained/xgb/model_{mdl_hex}.json"
    model = None
    if mdl_path.exists():
        param = {"max_depth": cfg.max_depth,
                 "eta": cfg.eta,
                 "objective": cfg.objective,
                 "device": cfg.device,
                 "eval_metric": list(cfg.eval_metric),
                 "nthread": cfg.nthread,
                 "seed": cfg.seed}
        model = xgb.Booster(param)
        model.load_model(mdl_path)
    else:
        print("Trained model not found!")

    return model


def set_prediction_score_on_node(bdd, preds):
    it = 0
    for lidx, layer in enumerate(bdd):
        for node in layer:
            node["pred"] = float(preds[it])
            it += 1

    return bdd


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

    if is_connected is False:
        print(np.max([node["pred"] for node in layer]))

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
def calculate_path_resistance(paths, layers, threshold=0.5, round_upto=1):
    resistances = []
    for path in paths:
        resistance = 0
        for i, layer in enumerate(layers):
            pred_score = layer[path[i + 1]]["pred"]
            _resistance = 0 if np.round(pred_score, round_upto) >= threshold else threshold - pred_score
            resistance += _resistance
        resistances.append(resistance)

        # s1 = cl[path[1]]["score"]
        # r1 = 0 if np.round(s1, round_upto) >= threshold else threshold - s1
        #
        # s2 = nl[path[2]]["score"]
        # r2 = 0 if np.round(s2, round_upto) >= threshold else threshold - s2
        #
        # resistances.append(r1 + r2)

    return resistances


def stitch_layer(bdd, lidx, heuristic, lookahead, threshold=0.5, round_upto=1):
    pl = bdd[lidx - 1] if lidx > 0 else None
    layers = [bdd[i] for i in range(lidx, lidx + lookahead)]
    # cl, nl = bdd[lidx], bdd[lidx + 1]

    # If BDD is disconnected on the first layer, select both nodes.
    if pl is None:
        cl = layers[0]
        for node in cl:
            node["prev_pred"] = float(node["pred"])
            node["pred"] = float(threshold + 0.001)
    else:
        partial_paths = [[node_idx] for node_idx, node in enumerate(pl) if
                         np.round(node["score"], round_upto) >= threshold]

        if heuristic == "naive":
            # min_resistance_idx = None
            # min_resistance = np.inf
            # for node in cl:
            #     for parent_idx in node["op"]:
            #         for path in partial_paths:
            #             if path[0] == parent_idx:
            #
            #
            #     for parent_idx in node["zp"]:
            #         pass
            pass

        elif heuristic == "min_resistance":
            print("Min resistance")
            # partial_paths = extend_paths(cl, partial_paths)
            # paths = extend_paths(nl, partial_paths)
            for i in range(lookahead - 1):
                partial_paths = extend_paths(layers[i], partial_paths)
            paths = extend_paths(layers[-1], partial_paths)
            # resistances = calculate_path_resistance(paths, cl, nl, threshold=threshold, round_upto=round_upto)
            resistances = calculate_path_resistance(paths, layers, threshold=threshold, round_upto=round_upto)

            resistances, paths = zip(*sorted(zip(resistances, paths), key=lambda x: x[0]))
            k = 1
            for r in resistances[1:]:
                if r > resistances[0]:
                    break
                else:
                    k += 1
            print("Selected paths: ", k)

            for path in paths[:k]:
                if np.round(cl[path[1]]["pred"], round_upto) < threshold:
                    cl[path[1]]["prev_pred"] = float(cl[path[1]]["pred"])
                    cl[path[1]]["pred"] = float(threshold + 0.001)

                if np.round(nl[path[2]]["pred"], round_upto) < threshold:
                    nl[path[2]]["prev_pred"] = float(nl[path[2]]["pred"])
                    nl[path[2]]["pred"] = float(threshold + 0.001)

    print(np.max([node["pred"] for node in bdd[lidx]]))

    return bdd


def get_pareto_states_per_layer(bdd):
    pareto_states_per_layer = []
    for layer in bdd:
        _pareto_states = []
        for node in layer:
            if np.round(node["pred"], 1) >= 0.5:
                _pareto_states.append(node["s"][0])

        pareto_states_per_layer.append(_pareto_states)

    return pareto_states_per_layer


def worker(rank, cfg, mdl_hex):
    env = libbddenvv1.BDDEnv()
    scorer = BinaryStatScores()
    model = load_model(cfg, mdl_hex)

    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    pids, bdd_data = [], []
    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        # Read instance
        inst_data = get_instance_data(cfg.prob.name, cfg.prob.size, cfg.deploy.split, pid)
        order = get_order(cfg.prob.name, cfg.deploy.order_type, inst_data)

        # Load BDD
        archive = resource_path / f"bdds/{cfg.prob.name}/{cfg.prob.size}.zip"
        file = f"{cfg.prob.size}/{cfg.deploy.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.deploy.label)

        # Get BDD data
        features, labels = convert_bdd_to_xgb_data_deploy(cfg.prob.name,
                                                          bdd=bdd,
                                                          inst_data=inst_data,
                                                          order=order,
                                                          state_norm_const=cfg.prob.state_norm_const,
                                                          layer_norm_const=cfg.prob.layer_norm_const)
        pos_idx = np.round(labels, 1) >= cfg.deploy.threshold
        labels_quantized = np.zeros_like(labels)
        labels_quantized[pos_idx] = 1

        # Predict
        dfeatures = xgb.DMatrix(features)
        preds = model.predict(dfeatures, iteration_range=(0, model.best_iteration + 1))

        pos_idx = np.round(preds, 1) >= cfg.deploy.threshold
        preds_quantized = np.zeros_like(preds)
        preds_quantized[pos_idx] = 1

        # Get prediction stats
        layers = np.round(features[:, -1] * cfg.prob.layer_norm_const)
        for l in list(map(int, np.unique(layers))):
            layer_nodes = layers == l

            labels_layer = labels_quantized[layer_nodes]
            preds_layer = preds_quantized[layer_nodes]

            tp, fp, tn, fn, sup = scorer(torch.from_numpy(preds_layer),
                                         torch.from_numpy(labels_layer))
            pred_stats_per_layer[l][1] += tp
            pred_stats_per_layer[l][2] += fp
            pred_stats_per_layer[l][3] += tn
            pred_stats_per_layer[l][4] += fn

        bdd = set_prediction_score_on_node(bdd, preds)
        # Check connectedness and perform stitching if necessary
        was_disconnected = False
        for lidx, layer in enumerate(bdd):
            prev_layer = bdd[lidx - 1] if lidx > 0 else None
            is_connected = check_connectedness(prev_layer,
                                               layer,
                                               threshold=cfg.deploy.threshold,
                                               round_upto=cfg.deploy.round_upto)
            if not is_connected:
                print("Disconnected, layer: ", lidx)
                was_disconnected = True
                bdd = stitch_layer(bdd, lidx, cfg.deploy.stitching_heuristic,
                                   threshold=cfg.deploy.threshold,
                                   round_upto=cfg.deploy.round_upto)

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
        pareto_states = get_pareto_states_per_layer(bdd)
        env.compute_pareto_frontier_with_pruning(pareto_states)
        _sol = {"x": env.x_sol,
                "z": env.z_sol,
                "ot": cfg.deploy.order_type}
        _data = [was_disconnected,
                 env.initial_node_count,
                 env.reduced_node_count,
                 env.initial_arcs_count,
                 env.reduced_arcs_count,
                 env.num_comparisons,
                 env.time_result]

        print(_data, len(_sol["x"]))
        # Save predicted BDD
        pred_bdd_path = resource_path / f"predictions/xgb/{mdl_hex}/pred_bdd"
        pred_bdd_path.mkdir(parents=True, exist_ok=True)
        pred_bdd_path = pred_bdd_path / f"{pid}.json"
        with open(pred_bdd_path, "w") as fp:
            json.dump(bdd, fp)

        pids.append(pid)
        bdd_data.append(_data)
        print("Finished processing ", pid)

    return pids, pred_stats_per_layer, bdd_data


@hydra.main(version_base="1.2", config_path="./configs", config_name="deploy_xgb.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()

    # Deploy model
    # pool = mp.Pool(processes=cfg.deploy.num_processes)
    # results = []
    # for rank in range(cfg.deploy.num_processes):
    #     results.append(pool.apply_async(worker, args=(rank, cfg, mdl_hex)))

    results = worker(0, cfg, mdl_hex)

    # # Fetch results
    # results = [r.get() for r in results]
    # pids, bdd_data = [], []
    # pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    # pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    # for r in results:
    #     pids.extend(r[0])
    #     pred_stats_per_layer[:, 1:] += r[1][:, 1:]
    #     bdd_data.extend(r[2])
    #
    # # Save results
    # df = pd.DataFrame(pred_stats_per_layer, columns=["layer", "tp", "fp", "tn", "fn"])
    # df.to_csv(resource_path / f"predictions/xgb/{mdl_hex}/connected_stats_per_layer_{cfg.deploy.split}.csv",
    #           index=False)
    #
    # bdd_stats, disconnected_layers = [], []
    # for idx, data in enumerate(bdd_data):
    #     if data[0] is None:
    #         inc, rnc, iac, rac, num_comparisons, sol, _time = data[1:]
    #         pid = pids[idx]
    #
    #         sol_pred_path = resource_path / f"predictions/xgb/{mdl_hex}/sols_pred"
    #         sol_pred_path.mkdir(exist_ok=True, parents=True)
    #         sol_path = sol_pred_path / f"sol_{pid}.json"
    #         with open(sol_path, "w") as fp:
    #             json.dump(sol, fp)
    #         time_path = sol_pred_path / f"time_{pid}.json"
    #         with open(time_path, "w") as fp:
    #             json.dump(_time, fp)
    #
    #         bdd_stats.append([cfg.prob.size,
    #                           pid,
    #                           cfg.deploy.split,
    #                           len(sol["x"]),
    #                           inc,
    #                           rnc,
    #                           iac,
    #                           rac,
    #                           num_comparisons,
    #                           _time["compilation"],
    #                           _time["reduction"],
    #                           _time["pareto"]])
    #     else:
    #         disconnected_layers.append(data[0])
    #
    # print("Disconnected instances: ", len(disconnected_layers))
    # if len(disconnected_layers):
    #     print("Average of disconnected layer: ", np.mean(disconnected_layers))
    # df = pd.DataFrame(bdd_stats, columns=["size",
    #                                       "pid",
    #                                       "split",
    #                                       "pred_nnds",
    #                                       "pred_inc",
    #                                       "pred_rnc",
    #                                       "pred_iac",
    #                                       "pred_rac",
    #                                       "pred_num_comparisons",
    #                                       "pred_compile",
    #                                       "pred_reduce",
    #                                       "pred_pareto"])
    # df.to_csv(resource_path / f"predictions/xgb/{mdl_hex}/pred_result_{cfg.deploy.split}.csv", index=False)


if __name__ == '__main__':
    main()
