import datetime
import hashlib
import io
import json
import os
import zipfile

import hydra
import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

from morbdd import resource_path


def get_xgb_model_name(max_depth=None,
                       eta=None,
                       min_child_weight=None,
                       subsample=None,
                       colsample_bytree=None,
                       objective=None,
                       num_round=None,
                       early_stopping_rounds=None,
                       evals=None,
                       eval_metric=None,
                       seed=None,
                       prob_name=None,
                       num_objs=None,
                       num_vars=None,
                       order=None,
                       layer_norm_const=None,
                       state_norm_const=None,
                       train_from_pid=None,
                       train_to_pid=None,
                       train_neg_pos_ratio=None,
                       train_min_samples=None,
                       train_flag_layer_penalty=None,
                       train_layer_penalty=None,
                       train_flag_imbalance_penalty=None,
                       train_flag_importance_penalty=None,
                       train_penalty_aggregation=None,
                       val_from_pid=None,
                       val_to_pid=None,
                       val_neg_pos_ratio=None,
                       val_min_samples=None,
                       val_flag_layer_penalty=None,
                       val_layer_penalty=None,
                       val_flag_imbalance_penalty=None,
                       val_flag_importance_penalty=None,
                       val_penalty_aggregation=None,
                       device=None):
    def get_model_name_knapsack():
        name = ""
        if max_depth is not None:
            name += f"{max_depth}-"
        if eta is not None:
            name += f"{eta}-"
        if min_child_weight is not None:
            name += f"{min_child_weight}-"
        if subsample is not None:
            name += f"{subsample}-"
        if colsample_bytree is not None:
            name += f"{colsample_bytree}-"
        if objective is not None:
            name += f"{objective}-"
        if num_round is not None:
            name += f"{num_round}-"
        if early_stopping_rounds is not None:
            name += f"{early_stopping_rounds}-"
        if type(evals) is list and len(evals):
            for eval in evals:
                name += f"{eval}"
        if type(eval_metric) is list and len(eval_metric):
            for em in eval_metric:
                name += f"{em}-"
        if seed is not None:
            name += f"{seed}"

        if prob_name is not None:
            name += f"{prob_name}-"
        if num_objs is not None:
            name += f"{num_objs}-"
        if num_vars is not None:
            name += f"{num_vars}-"
        if order is not None:
            name += f"{order}-"
        if layer_norm_const is not None:
            name += f"{layer_norm_const}-"
        if state_norm_const is not None:
            name += f"{state_norm_const}-"

        if train_from_pid is not None:
            name += f"{train_from_pid}-"
        if train_to_pid is not None:
            name += f"{train_to_pid}-"
        if train_neg_pos_ratio is not None:
            name += f"{train_neg_pos_ratio}-"
        if train_min_samples is not None:
            name += f"{train_min_samples}-"
        if train_flag_layer_penalty is not None:
            name += f"{train_flag_layer_penalty}-"
        if train_layer_penalty is not None:
            name += f"{train_layer_penalty}-"
        if train_flag_imbalance_penalty is not None:
            name += f"{train_flag_imbalance_penalty}-"
        if train_flag_importance_penalty is not None:
            name += f"{train_flag_importance_penalty}-"
        if train_penalty_aggregation is not None:
            name += f"{train_penalty_aggregation}-"

        if val_from_pid is not None:
            name += f"{val_from_pid}-"
        if val_to_pid is not None:
            name += f"{val_to_pid}-"
        if val_neg_pos_ratio is not None:
            name += f"{val_neg_pos_ratio}-"
        if val_min_samples is not None:
            name += f"{val_min_samples}-"
        if val_flag_layer_penalty is not None:
            name += f"{val_flag_layer_penalty}-"
        if val_layer_penalty is not None:
            name += f"{val_layer_penalty}-"
        if val_flag_imbalance_penalty is not None:
            name += f"{val_flag_imbalance_penalty}-"
        if val_flag_importance_penalty is not None:
            name += f"{val_flag_importance_penalty}-"
        if val_penalty_aggregation is not None:
            name += f"{val_penalty_aggregation}-"
        if device is not None:
            name += f"{device}"

        return name

    if prob_name == "knapsack":
        return get_model_name_knapsack()
    else:
        raise ValueError("Invalid problem!")


class Iterator(xgb.DataIter):
    def __init__(self, problem, size, split, neg_pos_ratio, min_samples, sampling_type, weights_type, labels_type,
                 names):
        self.problem = problem
        self.size = size
        self.split = split
        self.neg_pos_ratio = neg_pos_ratio
        self.min_samples = min_samples
        self.sampling_type = sampling_type
        self.weights_type = weights_type
        self.labels_type = labels_type
        self.names = names

        self.zf_sampling_type = zipfile.ZipFile(
            resource_path / f"xgb_data/{self.problem}/{self.size}/{self.split}/{sampling_type}.zip")
        self.zf_labels_type = zipfile.ZipFile(
            resource_path / f"xgb_data/{self.problem}/{self.size}/{self.split}/labels/{labels_type}.zip")

        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self.names):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        _name = self.names[self._it]
        with self.zf_sampling_type.open(f"{self.sampling_type}/{_name}", "r") as fp:
            data = io.BytesIO(fp.read())
            x = np.load(data)

        with self.zf_sampling_type.open(f"{self.sampling_type}/{self.weights_type}/{_name}", "r") as fp:
            data = io.BytesIO(fp.read())
            wt = np.load(data)

        with self.zf_labels_type.open(f"{self.labels_type}/{_name}", "r") as fp:
            data = io.BytesIO(fp.read())
            y = np.load(data)

        input_data(data=x, label=y, weight=wt)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0


def get_iterator(cfg, sampling_type, weights_type, label_type, split):
    valid_names = [f"{i}.npy" for i in range(cfg[split].from_pid, cfg[split].to_pid)]

    zf_path = zipfile.Path(resource_path / f"xgb_data/knapsack/{cfg.prob.size}/{split}/{sampling_type}.zip")
    filenames = [p.name for p in zf_path.joinpath(f'{sampling_type}').iterdir() if p.name in valid_names]
    print("Iterator on ", split, ": len - ", len(filenames))
    it = Iterator(cfg.prob.name,
                  cfg.prob.size,
                  split,
                  cfg[split].neg_pos_ratio,
                  cfg[split].min_samples,
                  sampling_type,
                  weights_type,
                  label_type,
                  filenames)

    return it


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_xgb.yaml")
def main(cfg):
    sampling_type = f"npr{cfg.train.neg_pos_ratio}ms{cfg.train.min_samples}"
    weights_type = ""
    if cfg.train.flag_layer_penalty:
        weights_type += f"{cfg.train.layer_penalty}-"
    weights_type += "1-" if cfg.train.flag_imbalance_penalty else "0-"
    weights_type += "1-" if cfg.train.flag_importance_penalty else "0-"
    weights_type += cfg.train.penalty_aggregation
    dtrain = xgb.DMatrix(get_iterator(cfg, sampling_type, weights_type, cfg.label, "train"))

    sampling_type = f"npr{cfg.val.neg_pos_ratio}ms{cfg.val.min_samples}"
    weights_type = ""
    if cfg.val.flag_layer_penalty:
        weights_type += f"{cfg.val.layer_penalty}-"
    weights_type += "1-" if cfg.val.flag_imbalance_penalty else "0-"
    weights_type += "1-" if cfg.val.flag_importance_penalty else "0-"
    weights_type += cfg.val.penalty_aggregation
    dval = xgb.DMatrix(get_iterator(cfg, sampling_type, weights_type, cfg.label, "val"))

    print("Number of training samples: ", dtrain.num_row())
    print("Number of validation samples: ", dval.num_row())
    print("Setting up training...")
    evals_result = {}
    evals = []
    for eval in cfg.evals:
        if eval == "train":
            evals.append((dtrain, "train"))
        elif eval == "val":
            evals.append((dval, "val"))

    param = {"max_depth": cfg.max_depth,
             "eta": cfg.eta,
             "min_child_weight": cfg.min_child_weight,
             "subsample": cfg.subsample,
             "colsample_bytree": cfg.colsample_bytree,
             "objective": cfg.objective,
             "device": cfg.device,
             "eval_metric": list(cfg.eval_metric),
             "nthread": cfg.nthread,
             "seed": cfg.seed}

    print("Started training...")
    bst = xgb.train(param, dtrain,
                    num_boost_round=cfg.num_round,
                    evals=evals,
                    early_stopping_rounds=cfg.early_stopping_rounds,
                    evals_result=evals_result)

    # Get model name
    mdl_path = resource_path / f"pretrained/xgb/{cfg.prob.name}/{cfg.prob.size}"
    mdl_path.mkdir(parents=True, exist_ok=True)
    mdl_name = get_xgb_model_name(max_depth=cfg.max_depth,
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
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    hex = h.hexdigest()

    # Save config
    with open(mdl_path.joinpath(f"config_{hex}.yaml"), "w") as fp:
        OmegaConf.save(cfg, fp)

    # Save model
    bst.save_model(mdl_path.joinpath(f"model_{hex}.json"))

    # Save metrics
    json.dump(evals_result, open(mdl_path.joinpath(f"metrics_{hex}.json"), "w"))

    # Save summary
    summary_obj = {"timestamp": str(datetime.datetime.now()),
                   "mdl_hex": hex,
                   "best_iteration": bst.best_iteration,
                   "eval_metric": list(cfg.eval_metric)[-1]}
    summary_obj.update({em: evals_result["train"][em][bst.best_iteration] for em in cfg.eval_metric})
    summary_obj.update({em: evals_result["val"][em][bst.best_iteration] for em in cfg.eval_metric})

    summary_path = mdl_path.joinpath("summary.json")
    if summary_path.exists():
        summary_json = json.load(open(summary_path, "r"))
        summary_json.append(summary_obj)
        json.dump(summary_json, open(summary_path, "w"))
    else:
        json.dump([summary_obj], open(summary_path, "w"))


if __name__ == '__main__':
    main()
