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


@hydra.main(version_base="1.2", config_path="./configs", config_name="cfg.yaml")
def main(cfg):
    model = load_model(cfg)

    pred_stats_per_layer = np.zeros((cfg.prob.num_vars, 5))
    pred_stats_per_layer[:, 0] = np.arange(pred_stats_per_layer.shape[0])

    for pid in range(cfg.deploy.from_pid, cfg.deploy.to_pid):
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
        preds = model.predict(dfeatures)

        # Get prediction stats
        layers = features[:, -1] * cfg.prob.layer_norm_const
        pred_stats_per_layer = []
        for l in np.unique(layers):
            tp = (preds[(layers == l) & (labels > 0)] >= 0.5).sum()
            fp = (preds[(layers == l) & (labels <= 0)] >= 0.5).sum()
            tn = (preds[(layers == l) & (labels <= 0)] < 0.5).sum()
            fn = (preds[(layers == l) & (labels > 0)] < 0.5).sum()
            pred_stats_per_layer[l][1] += tp
            pred_stats_per_layer[l][2] += fp
            pred_stats_per_layer[l][3] += tn
            pred_stats_per_layer[l][4] += fn

        # Get layer wise pareto states
        it = 0
        for layer in bdd:
            for node in layer:
                node["pred"] = preds[it]
                it += 1

        # Run BDD builder
        pass


if __name__ == '__main__':
    main()
