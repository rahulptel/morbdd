import io
import json
import os
import zipfile

import hydra
import numpy as np
import xgboost as xgb
import hashlib
from laser import resource_path
from omegaconf import OmegaConf
import pandas as pd
import datetime
from laser.utils import get_xgb_model_name


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
    mdl_path = resource_path / "pretrained/xgb"
    mdl_path.mkdir(parents=True, exist_ok=True)
    mdl_name = get_xgb_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    hex = h.hexdigest()

    # Save config
    with open(f"{hex}.yaml", "w") as fp:
        OmegaConf.save(cfg, fp)

    # Save model
    bst.save_model(mdl_path.joinpath(f"model_{hex}.json"))

    # Save metrics
    json.dump(evals_result, open(mdl_path.joinpath(f"metrics_{hex}.json"), "w"))

    # Save summary
    summary_line = [datetime.datetime.now(), hex]
    for em in cfg.eval_metric:
        summary_line.append(evals_result["train"][em][bst.best_iteration])
    for em in cfg.eval_metric:
        summary_line.append(evals_result["val"][em][bst.best_iteration])
    summary_line_str = ",".join(list(map(str, summary_line)))
    summary_path = mdl_path.joinpath("summary.csv")
    if summary_path.exists():
        with summary_path.open("a") as fp:
            fp.write(summary_line_str)
    else:
        with summary_path.open("w") as fp:
            fp.write(summary_line_str)


if __name__ == '__main__':
    main()
