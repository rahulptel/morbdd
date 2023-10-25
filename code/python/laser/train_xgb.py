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


@hydra.main(version_base="1.2", config_path="configs", config_name="train_xgb")
def main(cfg):
    dtrain = xgb.DMatrix("train.buffer")
    dval = xgb.DMatrix("val.buffer")
    dtest = None

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


if __name__ == '__main__':
    pass
