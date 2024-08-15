import json
import signal
import time

import numpy as np
import pandas as pd

from morbdd import CONST
from morbdd import ResourcePaths as path
from morbdd.baseline.baseline import Baseline
from morbdd.utils import MetricCalculator
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
from morbdd.utils import read_from_zip


class WidthRestrictedBDD(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metric_calculator = MetricCalculator(cfg.prob.n_vars)
        self.pred_pf = None
        self.orig_size = None
        self.rest_size = None
        self.size_ratio = None

    def get_env(self):
        libbddenv = __import__("libbddenvv2o" + str(self.cfg.prob.n_objs))
        env = libbddenv.BDDEnv()

        return env

    def load_pf(self, pid):
        pid = str(pid) + ".json"
        sol_path = path.sol / self.cfg.prob.name / self.cfg.prob.size / self.cfg.deploy.split / pid
        if sol_path.exists():
            print("Reading sol: ", sol_path)
            with open(sol_path, "r") as fp:
                sol = json.load(fp)
                return sol["z"]

        print("Sol path not found!")

    @staticmethod
    def compute_dd_size(dd):
        s = 0
        for l in dd:
            s += len(l)

        return s

    @staticmethod
    def node_select_random(idx_score, max_width, layer_width):
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
        selected_idx = [i[0] for i in idx_score[:max_width]]
        removed_idx = list(set(np.arange(layer_width)).difference(set(selected_idx)))

        return removed_idx

    def select_nodes(self, *args):
        pass

    def post_process(self, env, orig_dd, pid):
        restricted_dd = env.get_dd()
        self.orig_size = self.compute_dd_size(orig_dd)
        self.rest_size = self.compute_dd_size(restricted_dd)
        self.size_ratio = self.rest_size / self.orig_size
        true_pf = self.load_pf(pid)

        self.cardinality_raw, self.cardinality = -1, -1
        if true_pf is not None and self.pred_pf is not None:
            res = self.metric_calculator.compute_cardinality(true_pf, self.pred_pf)
            self.cardinality_raw = res['cardinality_raw']
            self.cardinality = res['cardinality']
            self.precision = res['precision']

    def save_result(self, pid, build_time, pareto_time):
        total_time = build_time + pareto_time

        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.split, pid, total_time, self.size_ratio, self.orig_size,
                            self.rest_size, self.cardinality, self.cardinality_raw, self.precision,
                            self.pred_pf.shape[0], build_time, pareto_time]],
                          columns=["size", "split", "pid", "total_time", "size_ratio", "orig_size", "rest_size",
                                   "cardinality", "cardinality_raw", "pred_precision", "n_pred_pf", "build_time",
                                   "pareto_time"])

        pid = str(pid) + ".csv"
        save_path = path.resource / "restricted_sols" / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split
        save_path = save_path / self.cfg.baseline.maxwidth
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / pid
        df.to_csv(save_path)


class KnapsackWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self, layer, max_width):
        if max_width < len(layer):
            if self.cfg.baseline.node_selection == "random":
                idx_score = [(i, n) for i, n in enumerate(layer)]
                return self.node_select_random(idx_score, max_width, len(layer))
            elif self.cfg.baseline.node_selection == "min_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                sorted(idx_score, key=lambda x: x[1])
                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                sorted(idx_score, key=lambda x: x[1], reverse=True)
                return [i[0] for i in idx_score[max_width:]]

        return []

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, 1, self.cfg.prob.n_objs, list(np.array(data["value"]).T),
                     [data["weight"]], [data["capacity"]])

    def worker(self, pid):
        archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(self.cfg.from_pid, self.cfg.to_pid):
            file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive, file, format="json")
            if bdd is not None:
                width = np.max([len(l) for l in bdd])
                max_width = int(width * self.cfg.baseline.max_width)

                # Load instance data
                data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)

                order = get_static_order("knapsack", "MinWt", data)
                env = self.get_env()
                env.reset(self.cfg.prob.problem_type,
                          self.cfg.prob.preprocess,
                          self.cfg.prob.pf_enum_method,
                          self.cfg.prob.maximization,
                          self.cfg.prob.dominance,
                          self.cfg.prob.bdd_type,
                          self.cfg.prob.maxwidth,
                          order)
                self.set_inst(env, data)
                env.preprocess_inst()

                # Initializes BDD with the root node
                env.initialize_dd_constructor()
                start = time.time()

                # Set the variable used to generate the next layer
                lid = 0
                # Restrict and build
                while lid < data["n_vars"] - 1:
                    env.generate_next_layer()
                    lid += 1

                    layer = env.get_layer(lid + 1)
                    removed_idx = self.select_nodes(layer, max_width)
                    if len(removed_idx):
                        env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)

                # Generate terminal layer
                env.generate_next_layer()
                build_time = time.time() - start

                env.reduce_dd()

                start = time.time()
                # Compute pareto frontier
                env.compute_pareto_frontier()
                pareto_time = time.time() - start
                try:
                    signal.alarm(1800)
                    self.pred_pf = env.get_frontier()["z"]
                except:
                    self.pred_pf = None
                signal.alarm(0)

                self.post_process(env, bdd, pid)
                self.save_result(pid, build_time, pareto_time)


class IndepsetWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self, layer, max_width):
        if max_width < len(layer):
            if self.cfg.baseline.node_selection == "random":
                idx_score = [(i, n) for i, n in enumerate(layer)]
                self.node_select_random(idx_score, max_width, len(layer))
            if self.cfg.baseline.node_selection == "min_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                sorted(idx_score, key=lambda x: x[1])
                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                sorted(idx_score, key=lambda x: x[1], reverse=True)
                return [i[0] for i in idx_score[max_width:]]

        return []

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, data["n_cons"], self.cfg.prob.n_objs, data["obj_coeffs"],
                     data["cons_coeffs"], data["rhs"])

    def worker(self, pid):
        archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(self.cfg.from_pid, self.cfg.to_pid):
            file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive, file, format="json")
            if bdd is not None:
                width = np.max([len(l) for l in bdd])
                max_width = int(width * self.cfg.baseline.max_width)

                # Load instance data
                data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)

                # Set BDD Manager
                order = []
                env = self.get_env()
                env.reset(self.cfg.prob.problem_type,
                          self.cfg.prob.preprocess,
                          self.cfg.prob.pf_enum_method,
                          self.cfg.prob.maximization,
                          self.cfg.prob.dominance,
                          self.cfg.prob.bdd_type,
                          self.cfg.prob.maxwidth,
                          order)
                self.set_inst(env, data)

                # Initializes BDD with the root node
                env.initialize_dd_constructor()
                start = time.time()

                # Set the variable used to generate the next layer
                env.set_var_layer(-1)
                lid = 0

                # Restrict and build
                while lid < data["n_vars"] - 1:
                    env.generate_next_layer()
                    env.set_var_layer(-1)
                    lid += 1

                    layer = env.get_layer(lid + 1)
                    removed_idx = self.select_nodes(layer, max_width)
                    if len(removed_idx):
                        env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)

                # Generate terminal layer
                env.generate_next_layer()
                build_time = time.time() - start

                start = time.time()
                # Compute pareto frontier
                env.compute_pareto_frontier()
                pareto_time = time.time() - start
                try:
                    signal.alarm(1800)
                    self.pred_pf = env.get_frontier()["z"]
                except:
                    self.pred_pf = None
                signal.alarm(0)

                self.post_process(env, bdd, pid)
                self.save_result(pid, build_time, pareto_time)
