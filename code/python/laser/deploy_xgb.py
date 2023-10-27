import hydra
import xgboost as xgb
from laser import resource_path
import numpy as np
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import get_featurizer
from laser.utils import MockConfig
from laser.utils import read_from_zip
from laser.utils import label_bdd
import pandas as pd
import multiprocessing as mp


def convert_bdd_to_xgb_data_deploy(problem,
                                   bdd=None,
                                   num_objs=None,
                                   num_vars=None,
                                   split=None,
                                   pid=None,
                                   order_type=None,
                                   state_norm_const=1000,
                                   layer_norm_const=100):
    size = f"{num_objs}_{num_vars}"

    def convert_bdd_to_xgb_data_deploy_knapsack():
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
        mdl_path = resource_path / "pretrained/xgb" / cfg.xgb.path
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
        raise ValueError("Invalid model name!")

    return model


def score_bdd_nodes_using_preds(bdd, preds):
    it = 0
    for lidx, layer in enumerate(bdd):
        for node in layer:
            node["pred"] = np.round(preds[it], 1)
            it += 1

    return bdd


def check_connectedness(pid, bdd):
    disconnected_layer = 2
    is_connected = False
    for lidx, layer in enumerate(bdd):
        is_connected = False
        for node in layer:
            if is_connected is False:
                if lidx == 0:
                    if node["pred"] >= 0.5:
                        is_connected = True
                        break
                else:
                    if node["pred"] >= 0.5:
                        if len(node["op"]):
                            prev_one = node["op"][0]
                            if bdd[lidx - 1][prev_one]["pred"] >= 0.5:
                                is_connected = True
                                break
                        if len(node["zp"]):
                            prev_zero = node["zp"][0]
                            if bdd[lidx - 1][prev_zero]["pred"] >= 0.5:
                                is_connected = True
                                break

        if is_connected is False:
            disconnected_layer = lidx + 1
            print(f"Instance {pid} is not connected!, Layer {disconnected_layer}")
            break

    return is_connected, disconnected_layer


def worker(rank, cfg):
    model = load_model(cfg)
    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    disconnected_layers = []
    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        print("Started processing ", pid)
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
                                                          num_objs=cfg.prob.num_objs,
                                                          num_vars=cfg.prob.num_vars,
                                                          split=cfg.deploy.split,
                                                          pid=pid,
                                                          order_type=cfg.deploy.order_type,
                                                          state_norm_const=cfg.prob.state_norm_const,
                                                          layer_norm_const=cfg.prob.layer_norm_const)

        # Predict
        dfeatures = xgb.DMatrix(features)
        preds = model.predict(dfeatures, iteration_range=(0, model.best_iteration + 1))

        # Get prediction stats
        layers = np.round(features[:, -1] * cfg.prob.layer_norm_const)
        for l in list(map(int, np.unique(layers))):
            tp = (np.round(preds[(layers == l) & (labels > 0)], 1) >= 0.5).sum()
            fp = (np.round(preds[(layers == l) & (labels <= 0)], 1) >= 0.5).sum()
            tn = (np.round(preds[(layers == l) & (labels <= 0)], 1) < 0.5).sum()
            fn = (np.round(preds[(layers == l) & (labels > 0)], 1) < 0.5).sum()
            pred_stats_per_layer[l][1] += tp
            pred_stats_per_layer[l][2] += fp
            pred_stats_per_layer[l][3] += tn
            pred_stats_per_layer[l][4] += fn

        bdd = score_bdd_nodes_using_preds(bdd, preds)
        is_connected, disconnected_layer = check_connectedness(pid, bdd)
        if not is_connected:
            disconnected_layers.append(disconnected_layer)

        # Run BDD builder
        # pass

    return pred_stats_per_layer, disconnected_layers


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):
    pool = mp.Pool(processes=cfg.deploy.num_processes)
    results = []
    for rank in range(cfg.deploy.num_processes):
        results.append(pool.apply_async(worker, args=(rank, cfg)))

    results = [r.get() for r in results]

    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])
    disconnected_layers = []
    for r in results:
        pred_stats_per_layer[:, 1:] += r[0][:, 1:]
        disconnected_layers.extend(r[1])

    print("Disconnected instances: ", len(disconnected_layers))
    print("Mean disconnected layer: ", np.mean(disconnected_layers))
    df = pd.DataFrame(pred_stats_per_layer, columns=["layer", "tp", "fp", "tn", "fn"])
    df.to_csv("stats_per_layer.csv", index=False)


if __name__ == '__main__':
    main()
