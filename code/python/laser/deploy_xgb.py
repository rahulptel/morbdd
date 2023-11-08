import json
import multiprocessing as mp

import hydra
import libbddenvv1
import numpy as np
import pandas as pd
import xgboost as xgb

from laser import resource_path
from laser.utils import FeaturizerConfig
from laser.utils import get_featurizer
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import label_bdd
from laser.utils import read_from_zip


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
        features["var"] = features["var"][order]
        num_var_features = features["var"].shape[1]

        features_lst, labels_lst = [], []
        for lidx, layer in enumerate(bdd):
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

                # Features
                features_lst.append(np.concatenate((inst_features,
                                                    _parent_var_feat,
                                                    _parent_node_feat,
                                                    features["var"][lidx],
                                                    _node_feat)))
                labels_lst.append(node["l"])

        return features_lst, labels_lst

    data = None
    if problem == "knapsack":
        data = convert_bdd_to_xgb_data_deploy_knapsack()

    assert data is not None
    return np.array(data[0]), np.array(data[1])


def load_model(cfg):
    if cfg.deploy.mdl == "xgb":
        mdl_path = resource_path / "pretrained/xgb" / cfg.mdl.path
        param = {"max_depth": cfg.mdl.max_depth,
                 "eta": cfg.mdl.eta,
                 "objective": cfg.mdl.objective,
                 "device": cfg.mdl.device,
                 "eval_metric": list(cfg.mdl.eval_metric),
                 "nthread": cfg.mdl.nthread,
                 "seed": cfg.seed}
        model = xgb.Booster(param)
        model.load_model(mdl_path)
    else:
        raise ValueError("Invalid model name!")

    return model


def score_bdd_nodes_using_preds(bdd, preds):
    it = 0
    for lidx, layer in enumerate(bdd):
        for node in layer:
            node["pred"] = np.round(preds[it], 1)
            it += 1

    return bdd


def check_connectedness(pid, bdd, threshold=0.5):
    disconnected_layer = 2
    is_connected = False
    for lidx, layer in enumerate(bdd):
        is_connected = False
        for node in layer:
            if is_connected is False:
                if lidx == 0:
                    if node["pred"] >= threshold:
                        is_connected = True
                        break
                else:
                    if node["pred"] >= threshold:
                        if len(node["op"]):
                            prev_one = node["op"][0]
                            if bdd[lidx - 1][prev_one]["pred"] >= threshold:
                                is_connected = True
                                break
                        if len(node["zp"]):
                            prev_zero = node["zp"][0]
                            if bdd[lidx - 1][prev_zero]["pred"] >= threshold:
                                is_connected = True
                                break

        if is_connected is False:
            disconnected_layer = lidx + 1
            print(f"Instance {pid} is not connected!, Layer {disconnected_layer}")
            break

    return is_connected, disconnected_layer


def get_pareto_states_per_layer(bdd):
    pareto_states_per_layer = []
    for layer in bdd:
        _pareto_states = []
        for node in layer:
            if np.round(node["pred"], 1) >= 0.5:
                _pareto_states.append(node["s"][0])

        pareto_states_per_layer.append(_pareto_states)

    return pareto_states_per_layer


def worker(rank, cfg):
    env = libbddenvv1.BDDEnv()
    model = load_model(cfg)
    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    pids, bdd_data = [], []

    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        pids.append(pid)
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

        # Predict
        dfeatures = xgb.DMatrix(features)
        preds = model.predict(dfeatures, iteration_range=(0, model.best_iteration + 1))

        # Get prediction stats
        layers = np.round(features[:, -1] * cfg.prob.layer_norm_const)
        for l in list(map(int, np.unique(layers))):
            tp = (np.round(preds[(layers == l) & (labels > 0)], 1) >= cfg.deploy.threshold).sum()
            fp = (np.round(preds[(layers == l) & (labels <= 0)], 1) >= cfg.deploy.threshold).sum()
            tn = (np.round(preds[(layers == l) & (labels <= 0)], 1) < cfg.deploy.threshold).sum()
            fn = (np.round(preds[(layers == l) & (labels > 0)], 1) < cfg.deploy.threshold).sum()
            pred_stats_per_layer[l][1] += tp
            pred_stats_per_layer[l][2] += fp
            pred_stats_per_layer[l][3] += tn
            pred_stats_per_layer[l][4] += fn

        bdd = score_bdd_nodes_using_preds(bdd, preds)
        is_connected, disconnected_layer = check_connectedness(pid, bdd, threshold=cfg.deploy.threshold)
        _data = []

        # Run BDD builder
        if is_connected:
            _data.append(None)
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
            _data.append(env.initial_node_count)
            _data.append(env.reduced_node_count)
            _data.append(env.initial_arcs_count)
            _data.append(env.reduced_arcs_count)
            _data.append(env.num_comparisons)
            _data.append({"x": env.x_sol,
                          "z": env.z_sol,
                          "ot": cfg.deploy.order_type})
            _data.append(env.time_result)
        else:
            _data.append(disconnected_layer)
            _data.extend([None] * 7)

        bdd_data.append(_data)

        print("Finished processing ", pid)

    return pids, pred_stats_per_layer, bdd_data


@hydra.main(version_base="1.2", config_path="./configs", config_name="deploy_xgb.yaml")
def main(cfg):
    # Deploy model
    pool = mp.Pool(processes=cfg.deploy.num_processes)
    results = []
    for rank in range(cfg.deploy.num_processes):
        results.append(pool.apply_async(worker, args=(rank, cfg)))

    # Fetch results
    results = [r.get() for r in results]
    pids, bdd_data = [], []
    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    for r in results:
        pids.extend(r[0])
        pred_stats_per_layer[:, 1:] += r[1][:, 1:]
        bdd_data.extend(r[2])

    # Save results
    df = pd.DataFrame(pred_stats_per_layer, columns=["layer", "tp", "fp", "tn", "fn"])
    df.to_csv("stats_per_layer.csv", index=False)

    bdd_stats, disconnected_layers = [], []
    for idx, data in enumerate(bdd_data):
        if data[0] is None:
            inc, rnc, iac, rac, num_comparisons, sol, _time = data[1:]
            pid = pids[idx]

            sol_pred_path = resource_path / f"sols_pred/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}"
            sol_pred_path.mkdir(exist_ok=True, parents=True)
            sol_path = sol_pred_path / f"sol_{pid}.json"
            with open(sol_path, "w") as fp:
                json.dump(sol, fp)
            time_path = sol_pred_path / f"time_{pid}.json"
            with open(time_path, "w") as fp:
                json.dump(_time, fp)

            bdd_stats.append([cfg.prob.size,
                              pid,
                              cfg.deploy.split,
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
            disconnected_layers.append(data[0])

    print("Disconnected instances: ", len(disconnected_layers))
    if len(disconnected_layers):
        print("Average of disconnected layer: ", np.mean(disconnected_layers))
    df = pd.DataFrame(bdd_stats, columns=["size",
                                          "pid",
                                          "split",
                                          "pred_nnds",
                                          "pred_inc",
                                          "pred_rnc",
                                          "pred_iac",
                                          "pred_rac",
                                          "pred_num_comparisons",
                                          "pred_compile",
                                          "pred_reduce",
                                          "pred_pareto"])
    df.to_csv("pred_result.csv", index=False)


if __name__ == '__main__':
    main()
