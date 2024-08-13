import datetime
import hashlib
import io
import json
import os
import zipfile

import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

from morbdd import resource_path
from .trainer import Trainer


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
        h = hashlib.blake2s(digest_size=32)
        h.update(self.mdl_name.encode("utf-8"))
        self.mdl_hex = h.hexdigest()

    def get_iterator(self, sampling_type, weights_type, split):
        valid_names = [f"{i}.npy" for i in range(self.cfg.dataset[split].from_pid, self.cfg.dataset[split].to_pid)]

        zf_path = zipfile.Path(resource_path / f"xgb_data/knapsack/{self.cfg.prob.size}/{split}/{sampling_type}.zip")
        filenames = [p.name for p in zf_path.joinpath(f'{sampling_type}').iterdir() if p.name in valid_names]
        print("Iterator on ", split, ": len - ", len(filenames))
        it = Iterator(self.cfg.prob.name,
                      self.cfg.prob.size,
                      split,
                      self.cfg.bdd_data.neg_to_pos_ratio,
                      self.cfg.bdd_data.min_samples,
                      sampling_type,
                      weights_type,
                      self.cfg.bdd_data.label,
                      filenames)

        return it

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
        mdl_path = self.mdl_path.joinpath(f"model_{self.mdl_hex}.json")
        print("Loading model: ", mdl_path, mdl_path.exists())
        if mdl_path.exists():
            self.bst = xgb.Booster(self.param)
            self.bst.load_model(mdl_path)

    def setup(self):
        sampling_type = f"npr{self.cfg.bdd_data.neg_to_pos_ratio}ms{self.cfg.bdd_data.min_samples}"
        weights_type = ""
        if self.cfg.bdd_data.flag_layer_penalty:
            weights_type += f"{self.cfg.bdd_data.layer_penalty}-"
        weights_type += "1-" if self.cfg.bdd_data.flag_imbalance_penalty else "0-"
        weights_type += "1-" if self.cfg.bdd_data.flag_importance_penalty else "0-"
        weights_type += self.cfg.bdd_data.penalty_aggregation
        # dtrain = xgb.DMatrix(get_iterator(cfg, sampling_type, weights_type, cfg.label, "train"))
        # dval = xgb.DMatrix(self.get_iterator(cfg, sampling_type, weights_type, cfg.label, "val"))
        self.set_dataset(sampling_type, weights_type, "train")
        self.set_dataset(sampling_type, weights_type, "val")
        print("Number of training samples: ", self.dtrain.num_row())
        print("Number of validation samples: ", self.dval.num_row())
        print("Setting up training...")

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

    def set_dataset(self, sampling_type, weights_type, split):
        setattr(self, f"d{split}", xgb.DMatrix(self.get_iterator(sampling_type, weights_type, split)))

    def set_optimizer(self):
        pass

    def get_trainer_str(self):
        name = ""
        if self.cfg.model.max_depth is not None:
            name += f"{self.cfg.model.max_depth}-"
        if self.cfg.model.eta is not None:
            name += f"{self.cfg.model.eta}-"
        if self.cfg.model.min_child_weight is not None:
            name += f"{self.cfg.model.min_child_weight}-"
        if self.cfg.model.subsample is not None:
            name += f"{self.cfg.model.subsample}-"
        if self.cfg.model.colsample_bytree is not None:
            name += f"{self.cfg.model.colsample_bytree}-"
        if self.cfg.model.objective is not None:
            name += f"{self.cfg.model.objective}-"
        if self.cfg.model.num_round is not None:
            name += f"{self.cfg.model.num_round}-"
        if self.cfg.model.early_stopping_rounds is not None:
            name += f"{self.cfg.model.early_stopping_rounds}-"
        if type(self.cfg.model.evals) is list and len(self.cfg.model.evals):
            for eval in self.cfg.model.evals:
                name += f"{eval}"
        if type(self.cfg.model.eval_metric) is list and len(self.cfg.model.eval_metric):
            for em in self.cfg.model.eval_metric:
                name += f"{em}-"
        if self.cfg.model.seed is not None:
            name += f"{self.cfg.model.seed}"

        if self.cfg.prob.name is not None:
            name += f"{self.cfg.prob.name}-"
        if self.cfg.prob.n_objs is not None:
            name += f"{self.cfg.prob.n_objs}-"
        if self.cfg.prob.n_vars is not None:
            name += f"{self.cfg.prob.n_vars}-"
        if self.cfg.prob.order_type is not None:
            name += f"{self.cfg.prob.order_type}-"
        if self.cfg.prob.layer_norm_const is not None:
            name += f"{self.cfg.prob.layer_norm_const}-"
        if self.cfg.prob.state_norm_const is not None:
            name += f"{self.cfg.prob.state_norm_const}-"

        if self.cfg.dataset.train.from_pid is not None:
            name += f"{self.cfg.dataset.train.from_pid}-"
        if self.cfg.dataset.train.to_pid is not None:
            name += f"{self.cfg.dataset.train.to_pid}-"
        if self.cfg.bdd_data.neg_to_pos_ratio is not None:
            name += f"{self.cfg.bdd_data.neg_to_pos_ratio}-"
        if self.cfg.bdd_data.min_samples is not None:
            name += f"{self.cfg.bdd_data.min_samples}-"
        if self.cfg.bdd_data.flag_layer_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_layer_penalty}-"
        if self.cfg.bdd_data.layer_penalty is not None:
            name += f"{self.cfg.bdd_data.layer_penalty}-"
        if self.cfg.bdd_data.flag_imbalance_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_imbalance_penalty}-"
        if self.cfg.bdd_data.flag_importance_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_importance_penalty}-"
        if self.cfg.bdd_data.penalty_aggregation is not None:
            name += f"{self.cfg.bdd_data.penalty_aggregation}-"

        if self.cfg.dataset.val.from_pid is not None:
            name += f"{self.cfg.dataset.val.from_pid}-"
        if self.cfg.dataset.val.to_pid is not None:
            name += f"{self.cfg.dataset.val.to_pid}-"
        if self.cfg.bdd_data.neg_to_pos_ratio is not None:
            name += f"{self.cfg.bdd_data.neg_to_pos_ratio}-"
        if self.cfg.bdd_data.min_samples is not None:
            name += f"{self.cfg.bdd_data.min_samples}-"
        if self.cfg.bdd_data.flag_layer_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_layer_penalty}-"
        if self.cfg.bdd_data.layer_penalty is not None:
            name += f"{self.cfg.bdd_data.layer_penalty}-"
        if self.cfg.bdd_data.flag_imbalance_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_imbalance_penalty}-"
        if self.cfg.bdd_data.flag_importance_penalty is not None:
            name += f"{self.cfg.bdd_data.flag_importance_penalty}-"
        if self.cfg.bdd_data.penalty_aggregation is not None:
            name += f"{self.cfg.bdd_data.penalty_aggregation}-"
        if self.cfg.device is not None:
            name += f"{self.cfg.device}"
        return name

    def save(self):
        # Save metrics
        json.dump(self.evals_result, open(self.mdl_path.joinpath(f"metrics_{hex}.json"), "w"))

        # Save summary
        summary_obj = {"timestamp": str(datetime.datetime.now()),
                       "mdl_hex": hex,
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
                             early_stopping_rounds=self.cfg.model.cfg.early_stopping_rounds,
                             evals_result=self.evals_result)

        # Save config
        with open(self.mdl_path.joinpath(f"config_{hex}.yaml"), "w") as fp:
            OmegaConf.save(self.cfg, fp)

        # Save model
        self.bst.save_model(self.mdl_path.joinpath(f"model_{hex}.json"))

    def setup_predict(self):
        self.set_mdl_param()
        self.set_model()

    def predict(self, features):
        return self.bst.predict(features, iteration_range=(0, self.bst.best_iteration + 1))
