import hashlib
import json
import multiprocessing as mp

import hydra
import numpy as np
import xgboost as xgb

from laser import resource_path
from laser.utils import FeaturizerConfig
from laser.utils import extract_node_features
from laser.utils import get_featurizer
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import get_xgb_model_name
from laser.utils import read_from_zip
import time
import pandas as pd


def call_get_model_name(cfg):
    return get_xgb_model_name(max_depth=cfg.max_depth,
                              eta=cfg.eta,
                              min_child_weight=cfg.min_child_weight,
                              subsample=cfg.subsample,
                              colsample_bytree=cfg.colsample_bytree,
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

        features_lst = []
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
                # labels_lst.append(node["l"])

        return features_lst

    features = None
    if problem == "knapsack":
        features = convert_bdd_to_xgb_data_deploy_knapsack()

    assert features is not None
    return np.array(features)


def load_model(cfg, mdl_hex):
    mdl_path = resource_path / f"pretrained/xgb/{cfg.prob.name}/{cfg.prob.size}/model_{mdl_hex}.json"
    model = None
    if mdl_path.exists():
        param = {"max_depth": cfg.max_depth,
                 "min_child_weight": cfg.min_child_weight,
                 "subsample": cfg.subsample,
                 "colsample_bytree": cfg.colsample_bytree,
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


def save_bdd(problem, size, split, pid, bdd, mdl_hex):
    pred_bdd_path = resource_path / f"predictions/xgb/{problem}/{size}/{split}/{mdl_hex}/pred_bdd"
    pred_bdd_path.mkdir(parents=True, exist_ok=True)
    pred_bdd_path = pred_bdd_path / f"{pid}.json"
    with open(pred_bdd_path, "w") as fp:
        json.dump(bdd, fp)


def save_time_result(r, problem, size, split, mdl_hex):
    df = pd.DataFrame(r, columns=["size", "split", "pid", "order_type", "time_featurize", "time_predict",
                                  "time_set_score"])
    df.to_csv(resource_path / f"predictions/xgb/{problem}/{size}/{split}/{mdl_hex}/time_pred_result.csv", index=False)


def worker(rank, cfg, mdl_hex):
    model = load_model(cfg, mdl_hex)
    time_result = []
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

        # Get BDD data
        time_featurize = time.time()
        features = convert_bdd_to_xgb_data_deploy(cfg.prob.name,
                                                  bdd=bdd,
                                                  inst_data=inst_data,
                                                  order=order,
                                                  state_norm_const=cfg.prob.state_norm_const,
                                                  layer_norm_const=cfg.prob.layer_norm_const)

        # Predict
        dfeatures = xgb.DMatrix(features)
        time_featurize = time.time() - time_featurize

        time_prediction = time.time()
        preds = model.predict(dfeatures, iteration_range=(0, model.best_iteration + 1))
        time_prediction = time.time() - time_prediction

        time_set_score = time.time()
        bdd = set_prediction_score_on_node(bdd, preds)
        time_set_score = time.time() - time_set_score

        save_bdd(cfg.prob.name, cfg.prob.size, cfg.deploy.split, pid, bdd, mdl_hex)
        print("Processed: ", pid)
        time_result.append([cfg.prob.size, cfg.deploy.split, pid, cfg.deploy.order_type,
                            time_featurize, time_prediction, time_set_score])

    return time_result


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

    # Fetch results
    results = [r.get() for r in results]
    r = []
    for result in results:
        r.extend(result)

    save_time_result(r, cfg.prob.name, cfg.prob.size, cfg.deploy.split, mdl_hex)
    # worker(0, cfg, mdl_hex)


if __name__ == '__main__':
    main()
