import numpy as np

from morbdd import ResourcePaths as path
from .dm import DataManager
from morbdd.utils.kp import get_instance_path
import multiprocessing as mp
from morbdd.utils.kp import get_instance_data
from morbdd.utils import read_from_zip


class KnapsackDataManager(DataManager):
    def generate_instances(self):
        rng = np.random.RandomState(self.cfg.seed)

        for s in self.cfg.size:
            n_objs, n_vars = map(int, s.split("_"))
            start, end = 0, self.cfg.n_train
            for split in ["train", "val", "test"]:
                for pid in range(start, end):
                    inst_path = get_instance_path(self.cfg.seed, n_objs, n_vars, split, pid, name=self.cfg.name)
                    inst_data = self.generate_instance(rng, n_vars, n_objs)
                    self.save_instance(inst_path, inst_data)

                if split == "train":
                    start = self.cfg.n_train
                    end = start + self.cfg.n_val
                elif split == "val":
                    start = self.cfg.n_train + self.cfg.n_val
                    end = start + self.cfg.n_test

    def generate_instance(self, rng, n_vars, n_objs):
        data = {"value": [], "weight": [], "capacity": 0}

        # Cost
        data["weight"] = rng.randint(1, self.cfg.max_obj_coeff + 1, n_vars)

        # Value
        if self.cfg.inst_type == "uncorr":
            for _ in range(n_objs):
                _objective = []
                for i in range(n_vars):
                    data["value"].append(rng.randint(1, self.cfg.max_obj_coeff + 1, n_vars))
        else:
            for _ in range(n_objs):
                _objective = []
                for i in range(n_vars):
                    lb = max(1, data["weight"][i] - (self.cfg.max_obj_coeff / 10))
                    ub = data["weight"][i] + (self.cfg.max_obj_coeff / 10)
                    _objective.append(rng.randint(lb, ub))
                data["value"].append(_objective)

        # Capacity
        data["capacity"] = np.ceil(0.5 * (np.sum(data["weight"])))

        return data

    def save_instance(self, inst_path, data):
        inst_path.parent.mkdir(parents=True, exist_ok=True)

        n_vars = len(list(data["weight"]))
        n_objs = len(data["value"])

        text = f"{n_vars}\n{n_objs}\n"
        for i in range(n_objs):
            string = " ".join([str(v) for v in data["value"][i]])
            text += string + "\n"
        string = " ".join([str(w) for w in data["weight"]])
        text += string + "\n"
        text += str(int(data["capacity"]))

        inst_path.open("w").write(text)

    def get_pareto_state_score(self, data, x, order=None):
        weight = data["cons_coeffs"][0]
        pareto_state_scores = []
        for i in range(1, x.shape[1]):
            x_partial = x[:, :i].reshape(-1, i)
            w_partial = weight[:i].reshape(i, 1)
            wt_dist = np.dot(x_partial, w_partial)
            pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
            pareto_score = pareto_score / pareto_score.sum()
            pareto_state_scores.append((pareto_state, pareto_score))

        return pareto_state_scores

    @staticmethod
    def preprocess_inst(env):
        env.preprocess_inst()

    def get_node_data(self, order, bdd):
        pass

    def get_instance_data(self, size, split, pid):
        return get_instance_data(size, split, pid)
