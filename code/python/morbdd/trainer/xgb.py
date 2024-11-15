import datetime
import hashlib
import io
import json
import os
import zipfile
from pprint import pprint

import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

from morbdd import ResourcePaths as path
from morbdd import resource_path
from morbdd.utils import get_dataset_prefix
from morbdd.utils import get_layer_weights
from .trainer import Trainer

import os

# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OMP_THREAD_LIMIT'] = '1'

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
            data = np.load(data)

            y = data[:, -1]
            scores = data[:, -2]
            lidxs = (data[:, -3]).astype(int)
            x = data[:, :-3]

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

class RankIterator(xgb.DataIter):
    def __init__(self, n_vars, zf_file, dataset_prefix, names):
        self.n_vars = n_vars
        self.zf_file = zf_file
        self.dataset_prefix = dataset_prefix
        self.names = names

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
            # Read data
            data = io.BytesIO(fp.read())
            data = np.load(data)
            # Prepare X
            X = data[:, :-2]
            # Prepare Y
            scores = data[:, -1]
            Y = (scores > 0).astype(np.int64)
            # Prepare group info for ranking task
            _, group = np.unique(data[:, -2], return_counts=True)
            input_data(data=X, label=Y, group=group)

        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

class XGBTrainer(Trainer):
    def __init__(self, cfg):
        Trainer.__init__(self, cfg)
        self.raw_train = None
        self.raw_val = None
        self.dtrain = None
        self.dval = None
        self.dtest = None
        self.pid_lid_map_train = None
        self.pid_lid_map_val = None
        self.evals = []
        self.bst = None
        self.params = {}
        self.evals_result = {}

        # Get model name
        self.mdl_path = resource_path / f"pretrained/{self.cfg.model.type}/{self.cfg.prob.name}/{self.cfg.prob.size}"
        self.mdl_path.mkdir(parents=True, exist_ok=True)
        self.mdl_name = self.get_trainer_str()

        # Convert to hex
        h = hashlib.blake2s(digest_size=32)
        h.update(self.mdl_name.encode("utf-8"))
        self.mdl_hex = h.hexdigest()

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

    def get_dataset_path(self):
        return path.dataset / f"{self.cfg.prob.name}/{self.cfg.model.type}/{self.cfg.prob.size}"

    def set_dataset_iterator(self, dataset_path):
        prefix = get_dataset_prefix(self.cfg.bdd_data.with_parent,
                                    self.cfg.layer_weight,
                                    self.cfg.bdd_data.neg_to_pos_ratio)
        prefix_zip = prefix + ".zip"

        # train_zip_path = zipfile.Path(dataset_path / "train" / prefix_zip)
        train_iterator = self.get_iterator("train", dataset_path, prefix)
        self.dtrain = xgb.DMatrix(train_iterator)

        # val_zip_path = zipfile.Path(dataset_path / "val" / prefix_zip)
        val_iterator = self.get_iterator("val", dataset_path, prefix)
        self.dval = xgb.DMatrix(val_iterator)

    def set_dataset_from_single_file(self, dataset_path):
        self.raw_train = np.load(dataset_path.joinpath("train.npy"))
        self.raw_train = self.raw_train.astype('float32')
        self.pid_lid_map_train = np.hstack((self.raw_train[:, 0].reshape(-1, 1),
                                            self.raw_train[:, -3].reshape(-1, 1)))
        assert np.min(self.pid_lid_map_train[:, 0]) == 0
        assert np.max(self.pid_lid_map_train[:, 0]) == 999

        data, label = self.raw_train[:, 1:-3], self.raw_train[:, -1]
        print("Train Min rank: ", np.min(label), ", Max rank: ", np.max(label))
        self.dtrain = xgb.DMatrix(data=data, label=label)
        self.raw_train, data, label = None, None, None

        self.raw_val = np.load(dataset_path.joinpath("val.npy"))
        self.raw_val = self.raw_val.astype('float32')
        self.pid_lid_map_val = np.hstack((self.raw_val[:, 0].reshape(-1, 1),
                                          self.raw_val[:, -3].reshape(-1, 1)))
        assert np.min(self.pid_lid_map_val[:, 0]) == 1000
        assert np.max(self.pid_lid_map_val[:, 0]) == 1099

        data, label = self.raw_val[:, 1:-3], self.raw_val[:, -1]
        print("Val Min rank: ", np.min(label), ", Max rank: ", np.max(label))
        self.dval = xgb.DMatrix(data=data, label=label)
        self.raw_val, data, label = None, None, None

    def set_dataset(self):
        dataset_path = self.get_dataset_path()
        if self.cfg.use_iterator:
            self.set_dataset_iterator(dataset_path)
        else:
            self.set_dataset_from_single_file(dataset_path)

        print("Number of training samples: ", self.dtrain.num_row())
        print("Number of validation samples: ", self.dval.num_row())

    def set_evals(self):
        self.evals = []
        for eval in self.cfg.model.evals:
            if eval == "train":
                self.evals.append((self.dtrain, "train"))
            elif eval == "val":
                self.evals.append((self.dval, "val"))

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
                      "seed": self.cfg.model.seed,
                      "random_state": self.cfg.model.seed}
        pprint(self.param)

    def set_model(self):
        self.set_mdl_param()
        mdl_path = self.mdl_path.joinpath(f"model_{self.mdl_name}.json")
        print("Loading model: ", mdl_path, ", Exists: ", mdl_path.exists())
        if mdl_path.exists():
            self.bst = xgb.Booster(self.param)
            self.bst.load_model(mdl_path)

    def setup(self):
        print("Setting up dataset...")
        self.set_dataset()
        self.set_evals()
        self.set_model()

    @staticmethod
    def print_stats(bst):
        s1 = [(k, v) for (k, v) in bst.get_score(importance_type='gain').items()]
        s1 = sorted(s1,
                    key=lambda x: x[1] / np.mean(list(bst.get_score(importance_type='gain').values())),
                    reverse=True)
        # s1 = [i[0] for i in s1]

        s2 = [(k, v) for (k, v) in bst.get_score(importance_type='weight').items()]
        s2 = sorted(s2,
                    key=lambda x: x[1] / np.mean(list(bst.get_score(importance_type='weight').values())),
                    reverse=True)
        # s2 = [i[0] for i in s2]

        s3 = [(k, v) for (k, v) in bst.get_score(importance_type='cover').items()]
        s3 = sorted(s3,
                    key=lambda x: x[1] / np.mean(list(bst.get_score(importance_type='cover').values())),
                    reverse=True)
        # s3 = [i[0] for i in s3]

        print("Gain Scores: ", s1)
        print("Weight Scores: ", s2)
        print("Cover score: ", s3)

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
        if self.cfg.model.type == "gbt" and self.cfg.model.objective != "binary:logistic":
            name += f"{self.cfg.model.objective}-"
        elif self.cfg.model.type == "gbt_rank" and self.cfg.model.objective != "rank:pairwise":
            name += f"{self.cfg.model.objective}-"

        if self.cfg.model.num_round != 250:
            name += f"ep{self.cfg.model.num_round}-"
        if self.cfg.model.early_stopping_rounds != 20:
            name += f"es{self.cfg.model.early_stopping_rounds}-"
        if self.cfg.model.evals[-1] != "val":
            name += f"eval{self.cfg.model.evals[-1]}-"
        if self.cfg.model.type == "gbt" and self.cfg.model.eval_metric[-1] != "logloss":
            name += f"l{self.cfg.model.eval_metric[-1]}"
        elif self.cfg.model.type == "gbt_rank" and self.cfg.model.eval_metric[-1] != "ndcg":
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
        print(summary_obj)
        summary_path = self.mdl_path.joinpath("summary.json")
        if summary_path.exists():
            summary_json = json.load(open(summary_path, "r"))
            summary_json.append(summary_obj)
            json.dump(summary_json, open(summary_path, "w"))
        else:
            json.dump([summary_obj], open(summary_path, "w"))

    def train(self):
        print("Started training...")
        self.bst = xgb.train(self.param,
                             self.dtrain,
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
        self.print_stats(self.bst)

    def setup_predict(self):
        self.set_mdl_param()
        self.set_model()

    def predict(self, features):
        return self.bst.predict(features, iteration_range=(0, self.bst.best_iteration + 1))

class XGBRankTrainer(XGBTrainer):
    def __init__(self, cfg):
        super(XGBRankTrainer, self).__init__(cfg)

    def get_iterator(self, split, dataset_path, dataset_prefix):
        dataset_prefix_zip = dataset_prefix + ".zip"

        zip_path = zipfile.Path(dataset_path / split / dataset_prefix_zip)
        zip_file = zipfile.ZipFile(dataset_path / split / dataset_prefix_zip)
        valid_names = [f"{i}.npy" for i in
                       range(self.cfg.dataset[split].from_pid, self.cfg.dataset[split].to_pid)]
        filenames = [p.name for p in zip_path.joinpath(dataset_prefix).iterdir() if p.name in valid_names]

        print("Iterator on ", split, ": len - ", len(filenames))

        return RankIterator(self.cfg.prob.n_vars,
                            zip_file,
                            dataset_prefix,
                            filenames)

    def get_groups(self, X):
        pids, layers = X[:, 0], X[:, 1]
        groups = []
        n_items, p_prev, l_prev = 1, pids[0], layers[0]
        for p, l in zip(pids[1:], layers[1:]):
            if p == p_prev and l == l_prev:
                n_items += 1
            elif p != p_prev:
                groups.append(n_items)
                p_prev = p
                n_items = 1
            elif l != l_prev:
                groups.append(n_items)
                l_prev = l
                n_items = 1
        groups.append(n_items)
        assert np.sum(groups) == X.shape[0]

        return groups


    def train(self):
        print("Started training...")
        self.dtrain.set_group(self.get_groups(self.pid_lid_map_train))
        self.dval.set_group(self.get_groups(self.pid_lid_map_val))
        super(XGBRankTrainer, self).train()