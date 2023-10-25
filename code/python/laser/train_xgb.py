import json
import zipfile

import pandas as pd
import numpy as np
import xgboost as xgb

from pathlib import Path
from operator import itemgetter
from laser.featurizer import KnapsackFeaturizer
import os
import hydra
from laser import resource_path


def get_dmatrix_filename(cfg, split):
    # file_path = resource_path / f"xgb_dataset/{cfg.prob}/{cfg.size}"
    name = f"{split}-{cfg[split].from_pid}-{cfg[split].to_pid}"
    if cfg.flag_layer_penalty:
        name += f"-{cfg.layer_penalty}"
    if cfg.flag_label_penalty:
        name += f"-{cfg.label_penalty}"
    name += ".buffer"

    return name


@hydra.main(version_base="1.2", config_path="./configs", config_name="train_xgb.yaml")
def main(cfg):
    dtrain = xgb.DMatrix(get_dmatrix_filename(cfg, "train"))
    dval = xgb.DMatrix(get_dmatrix_filename(cfg, "val"))
    dtest = None
    print(dtrain.num_row())

    evals = []
    for eval in cfg.evals:
        if eval == "train":
            evals.append((dtrain, "train"))

        elif eval == "val":
            evals.append((dval, "val"))

        elif eval == "test":
            if dtest is None:
                dtest = xgb.DMatrix("test.buffer")
            evals.append((dtest, "test"))

    param = {"max_depth": cfg.max_depth,
             "eta": cfg.eta,
             "evaluation_metric": cfg.evaluation_metric,
             "objective": cfg.objective,
             "device": cfg.device,
             "nthread": cfg.nthread}

    bst = xgb.train(param, dtrain,
                    num_boost_round=cfg.num_round,
                    evals=evals,
                    early_stopping_rounds=cfg.early_stopping_rounds)
    print(bst.best_iteration)

    bst.dump_model("xgb_model.txt")


if __name__ == '__main__':
    main()
