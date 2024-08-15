import datetime
import hashlib
import io
import json
import os
import zipfile

import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

from morbdd import ResourcePaths as path
from morbdd import resource_path
from morbdd.utils import get_dataset_prefix
from morbdd.utils import get_layer_weights
from .trainer import Trainer


class Iterator(xgb.DataIter):
    def __init__(self, n_vars, zf_file, dataset_prefix, flag_layer_penalty, layer_penalty_type,
                 flag_imbalance_penalty, flag_importance_penalty, penalty_aggregation, names):
        self.n_vars = n_vars
        self.zf_file = zf_file
        self.dataset_prefix = dataset_prefix
        self.flag_layer_penalty = flag_layer_penalty
        self.layer_penalty_type = layer_penalty_type
        self.flag_imbalance_penalty = flag_imbalance_penalty
        self.flag_importance_penalty = flag_importance_penalty
        self.penalty_aggregation = penalty_aggregation
        self.names = names

        if flag_layer_penalty:
            self.layer_penalty_store = get_layer_weights(self.flag_layer_penalty, self.layer_penalty_type, self.n_vars)
        else:
            self.layer_penalty_store = np.array([0] * (self.n_vars + 1))
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
        with self.zf_file.open(f"{self.dataset_prefix}/{_name}", "r") as fp:
            data = io.BytesIO(fp.read())
            x = np.load(data)
            scores = x[:, -1]
            lidxs = (x[:, -2]).astype(int)

            x = x[:, :-2]
            y = (scores > 0).astype(np.int64)
            wt = np.array(self.layer_penalty_store)[lidxs]
            if self.penalty_aggregation == "sum":
                if self.flag_importance_penalty:
                    wt += scores
            if np.sum(wt) == 0:
                wt = None

        input_data(data=x, label=y, weight=wt)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0


class XGBTrainer(Trainer):
    def __init__(self, cfg):
        Trainer.__init__(self, cfg)
        self.dtrain = None
        self.dval = None
        self.dtest = None
        self.evals = []
        self.bst = None
        self.params = {}
        self.evals_result = {}

        # Get model name
        self.mdl_path = resource_path / f"pretrained/xgb/{self.cfg.prob.name}/{self.cfg.prob.size}"
        self.mdl_path.mkdir(parents=True, exist_ok=True)
        self.mdl_name = self.get_trainer_str()

        # Convert to hex
        # h = hashlib.blake2s(digest_size=32)
        # h.update(self.mdl_name.encode("utf-8"))
        # self.mdl_hex = h.hexdigest()

    def get_iterator(self, split, dataset_path, dataset_prefix):
        dataset_prefix_zip = dataset_prefix + ".zip"

        zip_path = zipfile.Path(dataset_path / split / dataset_prefix_zip)
        zip_file = zipfile.ZipFile(dataset_path / split / dataset_prefix_zip)
        valid_names = [f"{i}.npy" for i in
                       range(self.cfg.dataset[split].from_pid, self.cfg.dataset[split].to_pid)]
        filenames = [p.name for p in zip_path.joinpath(dataset_prefix).iterdir() if p.name in valid_names]

        print("Iterator on ", split, ": len - ", len(filenames))

        return Iterator(self.cfg.prob.n_vars,
                        zip_file,
                        dataset_prefix,
                        self.cfg.bdd_data.flag_layer_penalty,
                        self.cfg.bdd_data.layer_penalty,
                        self.cfg.bdd_data.flag_imbalance_penalty,
                        self.cfg.bdd_data.flag_importance_penalty,
                        self.cfg.bdd_data.penalty_aggregation,
                        filenames)

    def set_mdl_param(self):
        self.param = {"max_depth": self.cfg.model.max_depth,
                      "eta": self.cfg.model.eta,
                      "min_child_weight": self.cfg.model.min_child_weight,
                      "subsample": self.cfg.model.subsample,
                      "colsample_bytree": self.cfg.model.colsample_bytree,
                      "objective": self.cfg.model.objective,
                      "device": self.cfg.device,
                      "eval_metric": list(self.cfg.model.eval_metric),
                      "nthread": self.cfg.model.nthread,
                      "seed": self.cfg.model.seed}

    def set_model(self):
        mdl_path = self.mdl_path.joinpath(f"model_{self.mdl_name}.json")
        print("Loading model: ", mdl_path, ", Exists: ", mdl_path.exists())
        if mdl_path.exists():
            self.bst = xgb.Booster(self.param)
            self.bst.load_model(mdl_path)

    def get_dataset_path(self):
        return path.dataset / f"{self.cfg.prob.name}/{self.cfg.model.type}/{self.cfg.prob.size}"

    def set_dataset(self, *args):
        dataset_path = self.get_dataset_path()
        prefix = get_dataset_prefix(self.cfg.bdd_data.with_parent, self.cfg.layer_weight,
                                    self.cfg.bdd_data.neg_to_pos_ratio)
        prefix_zip = prefix + ".zip"

        # train_zip_path = zipfile.Path(dataset_path / "train" / prefix_zip)
        train_iterator = self.get_iterator("train", dataset_path, prefix)
        self.dtrain = xgb.DMatrix(train_iterator)

        # val_zip_path = zipfile.Path(dataset_path / "val" / prefix_zip)
        val_iterator = self.get_iterator("val", dataset_path, prefix)
        self.dval = xgb.DMatrix(val_iterator)

        print("Number of training samples: ", self.dtrain.num_row())
        print("Number of validation samples: ", self.dval.num_row())

    def setup(self):
        print("Setting up dataset...")
        self.set_dataset()

        self.evals = []
        for eval in self.cfg.model.evals:
            if eval == "train":
                self.evals.append((self.dtrain, "train"))
            elif eval == "val":
                self.evals.append((self.dval, "val"))
        self.set_mdl_param()
        self.set_model()

    @staticmethod
    def print_stats(*args):
        pass

    def set_optimizer(self):
        pass

    def get_trainer_str(self):
        name = ""
        if self.cfg.model.max_depth != 5:
            name += f"d{self.cfg.model.max_depth}-"
        if self.cfg.model.eta != 0.3:
            name += f"eta{self.cfg.model.eta}-"
        if self.cfg.model.min_child_weight != 100:
            name += f"mcw{self.cfg.model.min_child_weight}-"
        if self.cfg.model.subsample != 1:
            name += f"ss{self.cfg.model.subsample}-"
        if self.cfg.model.colsample_bytree != 1:
            name += f"csbt{self.cfg.model.colsample_bytree}-"
        if self.cfg.model.objective != "binary:logistic":
            name += f"{self.cfg.model.objective}-"
        if self.cfg.model.num_round != 250:
            name += f"ep{self.cfg.model.num_round}-"
        if self.cfg.model.early_stopping_rounds != 20:
            name += f"es{self.cfg.model.early_stopping_rounds}-"
        if self.cfg.model.evals[-1] != "val":
            name += f"eval{self.cfg.model.evals[-1]}-"
        if self.cfg.model.eval_metric[-1] != "logloss":
            name += f"l{self.cfg.model.eval_metric[-1]}"
        if self.cfg.model.seed != 789541:
            name += f"seed{self.cfg.model.seed}"

        if self.cfg.dataset.train.from_pid != 0:
            name += f"trs{self.cfg.dataset.train.from_pid}-"
        if self.cfg.dataset.train.to_pid != 1000:
            name += f"tre{self.cfg.dataset.train.to_pid}-"
        if self.cfg.dataset.val.from_pid != 1000:
            name += f"vls{self.cfg.dataset.val.from_pid}-"
        if self.cfg.dataset.val.to_pid != 1100:
            name += f"vle{self.cfg.dataset.val.to_pid}-"

        if self.cfg.bdd_data.neg_to_pos_ratio != 1:
            name += f"npr{self.cfg.bdd_data.neg_to_pos_ratio}-"
        if self.cfg.bdd_data.layer_penalty != "exponential":
            name += f"lp{self.cfg.bdd_data.layer_penalty}-"
        if self.cfg.bdd_data.flag_importance_penalty is False:
            name += "nfimp-"
        if self.cfg.bdd_data.penalty_aggregation != "sum":
            name += f"{self.cfg.bdd_data.penalty_aggregation}-"
        if self.cfg.device != "cpu":
            name += f"dev{self.cfg.device}"
        return name

    def save(self):
        # Save metrics
        json.dump(self.evals_result, open(self.mdl_path.joinpath(f"metrics_{self.mdl_name}.json"), "w"))

        # Save summary
        summary_obj = {"timestamp": str(datetime.datetime.now()),
                       "mdl_hex": self.mdl_name,
                       "best_iteration": self.bst.best_iteration,
                       "eval_metric": list(self.cfg.model.eval_metric)[-1]}
        summary_obj.update({em: self.evals_result["train"][em][self.bst.best_iteration]
                            for em in self.cfg.model.eval_metric})
        summary_obj.update({em: self.evals_result["val"][em][self.bst.best_iteration]
                            for em in self.cfg.model.eval_metric})
        summary_path = self.mdl_path.joinpath("summary.json")
        if summary_path.exists():
            summary_json = json.load(open(summary_path, "r"))
            summary_json.append(summary_obj)
            json.dump(summary_json, open(summary_path, "w"))
        else:
            json.dump([summary_obj], open(summary_path, "w"))

    def train(self):
        print("Started training...")
        self.bst = xgb.train(self.param, self.dtrain,
                             num_boost_round=self.cfg.model.num_round,
                             evals=self.evals,
                             early_stopping_rounds=self.cfg.model.early_stopping_rounds,
                             evals_result=self.evals_result)

        # Save config
        with open(self.mdl_path.joinpath(f"config_{self.mdl_name}.yaml"), "w") as fp:
            OmegaConf.save(self.cfg, fp)

        # Save model
        self.bst.save_model(self.mdl_path.joinpath(f"model_{self.mdl_name}.json"))
        self.save()

    def setup_predict(self):
        self.set_mdl_param()
        self.set_model()

    def predict(self, features):
        return self.bst.predict(features, iteration_range=(0, self.bst.best_iteration + 1))
