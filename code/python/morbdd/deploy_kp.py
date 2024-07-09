import hashlib
import time

import hydra
import numpy as np
import xgboost as xgb

from morbdd import resource_path
from morbdd.utils import FeaturizerConfig
from morbdd.utils import LayerStitcher, LayerNodeSelector
from morbdd.utils import compute_cardinality
from morbdd.utils import compute_dd_size
from morbdd.utils import extract_node_features
from morbdd.utils import get_featurizer
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import get_xgb_model_name
from morbdd.utils import load_orig_dd
from morbdd.utils import load_pf

RESTRICT = 1
RELAX = 2


def call_get_model_name(cfg):
    if cfg.prob.name == "knapsack":
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
    else:
        raise ValueError("Invalid problem name!")


def get_layer_features(lidx, layer, prev_layer, inst_data, inst_features, var_features, num_var_features,
                       layer_norm_const=100, state_norm_const=1000):
    # Parent variable features
    _parent_var_feat = -1 * np.ones(num_var_features) \
        if lidx == 0 \
        else var_features[lidx - 1]
    _var_feat = var_features[lidx]

    features_lst = []
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

    return np.array(features_lst)


def load_model(cfg, mdl_hex):
    mdl_path = resource_path / f"pretrained/xgb/{cfg.prob.name}/{cfg.prob.size}/model_{mdl_hex}.json"
    model = None
    if mdl_path.exists():
        param = {"max_depth": cfg.xgb.max_depth,
                 "min_child_weight": cfg.xgb.min_child_weight,
                 "subsample": cfg.xgb.subsample,
                 "colsample_bytree": cfg.xgb.colsample_bytree,
                 "eta": cfg.xgb.eta,
                 "objective": cfg.xgb.objective,
                 "device": cfg.device,
                 "eval_metric": list(cfg.xgb.eval_metric),
                 "nthread": cfg.nthread,
                 "seed": cfg.seed}
        model = xgb.Booster(param)
        model.load_model(mdl_path)
    else:
        print("Trained model not found!")

    return model


def get_env(n_objs=3):
    libbddenv = __import__("libbddenvv2o" + str(n_objs))
    env = libbddenv.BDDEnv()

    return env


def worker(rank, cfg, mdl_hex):
    model = load_model(cfg, mdl_hex)
    env = get_env(n_objs=cfg.prob.n_objs)

    # Extract instance and variable features
    featurizer_conf = FeaturizerConfig()
    featurizer = get_featurizer("knapsack", featurizer_conf)

    node_selector = LayerNodeSelector(cfg.deploy.node_select.strategy,
                                      width=cfg.deploy.node_select.width,
                                      threshold=cfg.deploy.node_select.threshold)
    layer_stitcher = LayerStitcher()

    time_result = []
    for pid in range(cfg.deploy.from_pid + rank, cfg.deploy.to_pid, cfg.deploy.num_processes):
        # Read instance
        data = get_instance_data(cfg.prob.name, cfg.prob.size, cfg.deploy.split, pid)
        order = get_static_order(cfg.prob.name, cfg.deploy.order_type, data)

        features = featurizer.get(data)
        # Instance features
        inst_features = features["inst"][0]
        # Variable features. Reordered features based on ordering
        var_features = features["var"][order]
        num_var_features = var_features.shape[1]

        start = time.time()
        env.reset(cfg.problem_type,
                  cfg.preprocess,
                  cfg.method,
                  cfg.maximization,
                  cfg.dominance,
                  cfg.bdd_type,
                  cfg.maxwidth,
                  order)
        env.set_inst(cfg.prob.n_vars, data["n_cons"], cfg.prob.n_objs, data["obj_coeffs"], data["cons_coeffs"],
                     data["rhs"])
        # Initializes BDD with the root node
        env.initialize_dd_constructor()
        lid = 0

        while lid < data["n_vars"] - 1:
            env.generate_next_layer()
            lid += 1

            prev_layer = env.get_layer(lid - 1)
            layer = env.get_layer(lid)
            features = get_layer_features(lid, layer, prev_layer, data, inst_features, var_features,
                                          num_var_features, layer_norm_const=100, state_norm_const=1000)
            # Predict
            dfeatures = xgb.DMatrix(features)
            scores = model.predict(dfeatures, iteration_range=(0, model.best_iteration + 1))

            if cfg.deploy.node_select.prune_after < lid < cfg.deploy.node_select.prune_upto:
                selection, selected_idx, removed_idx = node_selector(lid, scores)
                # Stitch in case of a disconnected BDD
                if len(removed_idx) == len(scores):
                    removed_idx = layer_stitcher(scores)
                    print("Disconnected at layer: ", {lid})
                # Restrict if necessary
                if len(removed_idx):
                    env.approximate_layer(lid, RESTRICT, 1, removed_idx)

        # Generate terminal layer
        env.generate_next_layer()
        build_time = time.time() - start

        start = time.time()
        # Compute pareto frontier
        env.compute_pareto_frontier()
        pareto_time = time.time() - start

        orig_dd = load_orig_dd(cfg)
        restricted_dd = env.get_dd()
        orig_size = compute_dd_size(orig_dd)
        rest_size = compute_dd_size(restricted_dd)
        size_ratio = rest_size / orig_size

        true_pf = load_pf(cfg)
        try:
            pred_pf = env.get_frontier()["z"]
        except:
            pred_pf = None

        cardinality_raw, cardinality = -1, -1
        if true_pf is not None and pred_pf is not None:
            cardinality_raw = compute_cardinality(true_pf=true_pf, pred_pf=pred_pf)
            cardinality = cardinality_raw / len(true_pf)

        # run_path = get_run_path(cfg, helper.get_checkpoint_path().stem)
        # run_path.mkdir(parents=True, exist_ok=True)
        #
        # total_time = data_preprocess_time + node_emb_time + inst_emb_time + build_time + pareto_time
        # df = pd.DataFrame([[cfg.size, cfg.deploy.split, cfg.deploy.pid, total_time, size_ratio, cardinality,
        #                     rest_size, orig_size, cardinality_raw, data_preprocess_time, node_emb_time, inst_emb_time,
        #                     build_time, pareto_time]],
        #                   columns=["size", "split", "pid", "total_time", "size", "cardinality", "rest_size",
        #                            "orig_size",
        #                            "cardinality_raw", "data_preprocess_time", "node_emb_time", "inst_emb_time",
        #                            "build_time", "pareto_time"])
        # print(df)
        #
        # pid = str(cfg.deploy.pid) + ".csv"
        # result_path = run_path / pid
        # df.to_csv(result_path)


@hydra.main(version_base="1.2", config_path="./configs", config_name="deploy.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()
    print(f"Using model: {mdl_hex}")

    # Deploy model
    # pool = mp.Pool(processes=cfg.deploy.num_processes)
    # results = []
    # for rank in range(cfg.deploy.num_processes):
    #     results.append(pool.apply_async(worker, args=(rank, cfg, mdl_hex)))
    #
    # # Fetch results
    # results = [r.get() for r in results]
    # r = []
    # for result in results:
    #     r.extend(result)

    r = worker(0, cfg, mdl_hex)
    # save_time_result(r, cfg.prob.name, cfg.prob.size, cfg.deploy.split, mdl_hex)


if __name__ == '__main__':
    main()
