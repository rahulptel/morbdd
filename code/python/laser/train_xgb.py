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


class Iterator(xgb.DataIter):
    def __init__(self, problem, size, split, neg_pos_ratio, min_samples, names):
        self.problem = problem
        self.size = size
        self.split = split
        self.neg_pos_ratio = neg_pos_ratio
        self.min_samples = min_samples
        self.dtype = f"npr{self.neg_pos_ratio}ms{self.min_samples}"
        self.names = names

        self.zf = zipfile.ZipFile(resource_path / f"xgb_data/{self.problem}/{self.size}/{self.split}.zip")

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
        with self.zf.open(f"{self.split}/{self.dtype}/{_name}", "r") as fp:
            data = io.BytesIO(fp.read())
            data_np = np.load(data)
        input_data(data=data_np[:, :-2], label=data_np[:, -2], weight=data_np[:, -1])
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0


def get_dmatrix_filename(cfg, split):
    # file_path = resource_path / f"xgb_dataset/{cfg.prob}/{cfg.size}"
    name = f"{split}-{cfg[split].from_pid}-{cfg[split].to_pid}"
    if cfg.flag_layer_penalty:
        name += f"-{cfg.layer_penalty}"
    if cfg.flag_label_penalty:
        name += f"-{cfg.label_penalty}"
    name += ".buffer"

    return name


def get_iterator(cfg, dtype, split):
    zf_path = zipfile.Path(resource_path / f"xgb_data/knapsack/{cfg.prob.size}/{split}.zip")
    valid_names = [f"{i}.npy" for i in range(cfg[split].from_pid, cfg[split].to_pid)]
    filenames = [p.name for p in zf_path.joinpath(f'{split}/{dtype}').iterdir() if p.name in valid_names]
    print("Iterator on ", split, ": len - ", len(filenames))
    it = Iterator(cfg.prob.name,
                  cfg.prob.size,
                  split,
                  cfg[split].neg_pos_ratio,
                  cfg[split].min_samples,
                  filenames)

    return it


def get_xgb_model_name(cfg):
    def get_model_name_knapsack():
        name = f"{cfg.max_depth}-"
        name += f"{cfg.eta}-"
        name += f"{cfg.objective}-"
        name += f"{cfg.num_round}-"
        name += f"{cfg.early_stopping_rounds}-"
        for eval in cfg.evals:
            name += f"{eval}"
        for em in cfg.eval_metric:
            name += f"{em}-"
        name += f"{cfg.seed}"

        name += f"{cfg.prob.name}-"
        name += f"{cfg.prob.num_objs}-"
        name += f"{cfg.prob.num_vars}-"
        name += f"{cfg.prob.order}-"
        name += f"{cfg.prob.layer_norm_const}-"
        name += f"{cfg.prob.state_norm_const}-"

        name += f"{cfg.train.from_pid}-"
        name += f"{cfg.train.to_pid}-"
        name += f"{cfg.train.neg_pos_ratio}-"
        name += f"{cfg.train.min_samples}-"
        name += f"{cfg.train.flag_layer_penalty}-"
        name += f"{cfg.train.layer_penalty}-"
        name += f"{cfg.train.flag_imbalance_penalty}-"
        name += f"{cfg.train.flag_importance_penalty}-"
        name += f"{cfg.train.penalty_aggregation}-"

        name += f"{cfg.val.from_pid}-"
        name += f"{cfg.val.to_pid}-"
        name += f"{cfg.val.neg_pos_ratio}-"
        name += f"{cfg.val.min_samples}-"
        name += f"{cfg.val.flag_layer_penalty}-"
        name += f"{cfg.val.layer_penalty}-"
        name += f"{cfg.val.flag_imbalance_penalty}-"
        name += f"{cfg.val.flag_importance_penalty}-"
        name += f"{cfg.val.penalty_aggregation}-"

        name += f"{cfg.device}"

        return name

    if cfg.prob.name == "knapsack":
        return get_model_name_knapsack()
    else:
        raise ValueError("Invalid problem!")


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_xgb.yaml")
def main(cfg):
    dtype = f"npr{cfg.train.neg_pos_ratio}ms{cfg.train.min_samples}"
    dtrain = xgb.DMatrix(get_iterator(cfg, dtype, "train"))
    dval = xgb.DMatrix(get_iterator(cfg, dtype, "val"))

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
    mdl_path = resource_path / "pretrained/xgb"
    mdl_path.mkdir(parents=True, exist_ok=True)
    # Get model name
    mdl_name = get_xgb_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name)
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
        summary_line.append(evals_result[em][bst.best_iteration])
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
