import json
import random
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
        self.orig_width = None
        self.rest_width = None

    def get_env(self):
        libbddenv = __import__("libbddenvv2o" + str(self.cfg.prob.n_objs))
        env = libbddenv.BDDEnv()

        return env

    def set_inst(self, *args):
        raise NotImplementedError

    def load_pf(self, pid):
        pid = str(pid) + ".json"
        sol_path = path.sol / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / pid
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
        raise NotImplementedError

    def compute_dd_width(self, dd):
        return np.max([len(l) for l in dd])

    def post_process(self, env, orig_dd, pid):
        restricted_dd = env.get_dd()
        self.orig_size = self.compute_dd_size(orig_dd)
        self.rest_size = self.compute_dd_size(restricted_dd)
        self.size_ratio = self.rest_size / self.orig_size
        self.orig_width = self.compute_dd_width(orig_dd)
        self.rest_width = self.compute_dd_width(restricted_dd)
        true_pf = self.load_pf(pid)

        self.cardinality_raw, self.cardinality = -1, -1
        if true_pf is not None and self.pred_pf is not None:
            res = self.metric_calculator.compute_cardinality(true_pf, self.pred_pf)
            self.cardinality_raw = res['cardinality_raw']
            self.cardinality = res['cardinality']
            self.precision = res['precision']

    def save_result(self, seed, pid, build_time, pareto_time):
        total_time = build_time + pareto_time
        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.split, pid, total_time, self.size_ratio, self.orig_size,
                            self.rest_size, self.orig_width, self.rest_width, self.cardinality, self.cardinality_raw,
                            self.precision,
                            len(self.pred_pf), build_time, pareto_time]],
                          columns=["size", "split", "pid", "total_time", "size_ratio", "orig_size", "rest_size",
                                   "orig_width", "rest_width", "cardinality", "cardinality_raw", "pred_precision",
                                   "n_pred_pf", "build_time", "pareto_time"])

        pid = str(pid) + f"_{seed}.csv"
        save_path = path.resource / "restricted_sols" / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split
        save_path = save_path / self.cfg.baseline.node_selection / str(self.cfg.baseline.max_width)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / pid
        print(df)
        df.to_csv(save_path)

    def set_var_layer(self, env):
        pass

    def build_dd(self, env, max_width, rng):
        # Set the variable used to generate the next layer
        lid = 0
        self.set_var_layer(env)

        # Restrict and build
        while lid < self.cfg.prob.n_vars - 1:
            env.generate_next_layer()
            self.set_var_layer(env)
            lid += 1

            layer = env.get_layer(lid)
            removed_idx = self.select_nodes(rng, layer, max_width)
            if len(removed_idx):
                env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)

        # Generate terminal layer
        env.generate_next_layer()

    def reduce_dd(self, env):
        pass

    def run_pipeline(self, seed, pid, data, bdd, order, max_width):
        print(f"Pid: {pid}, seed: {seed}")
        rng = random.Random(seed)
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

        env.initialize_dd_constructor()
        start = time.time()
        self.build_dd(env, max_width, rng)
        build_time = time.time() - start
        self.reduce_dd(env)

        # Compute pareto frontier
        start = time.time()
        env.compute_pareto_frontier()
        pareto_time = time.time() - start
        try:
            signal.alarm(1800)
            self.pred_pf = env.get_frontier()["z"]
        except:
            self.pred_pf = None
        signal.alarm(0)

        self.post_process(env, bdd, pid)
        self.save_result(seed, pid, build_time, pareto_time)

    def worker(self, pid):
        signal.signal(signal.SIGALRM, handle_timeout)

        archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
        file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is not None:
            width = np.max([len(l) for l in bdd])
            max_width = int(width * (self.cfg.baseline.max_width / 100))
            print(f"Width: {width}, max width: {max_width}")

            # Load instance data
            data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, pid)

            order = get_static_order(self.cfg.prob.name, self.cfg.prob.order_type, data)
            print("Static Order: ", order)

            n_trails = 5 if self.cfg.baseline.node_selection == "random" else 1
            for t in range(n_trails):
                seed = self.cfg.trial_seeds[t]
                self.run_pipeline(seed, pid, data, bdd, order, max_width)


class KnapsackWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self, rng, layer, max_width):
        # print(f"Max width is {max_width}, layer width is {len(layer)}")
        n_layer = len(layer)
        if max_width < n_layer:
            # print("Restricting...")
            if self.cfg.baseline.node_selection == "random":
                idxs = np.arange(n_layer)
                rng.shuffle(idxs)

                return idxs[:max_width]
            elif self.cfg.baseline.node_selection == "min_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1])

                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

                return [i[0] for i in idx_score[max_width:]]

        return []

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, 1, self.cfg.prob.n_objs, list(np.array(data["value"]).T),
                     [data["weight"]], [data["capacity"]])

    def reduce_dd(self, env):
        print("Reducing dd...")
        env.reduce_dd()


class IndepsetWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self, rng, layer, max_width):
        n_layer = len(layer)
        if max_width < n_layer:
            if self.cfg.baseline.node_selection == "random":
                idxs = np.arange(n_layer)
                rng.shuffle(idxs)

                return idxs[:max_width]
            if self.cfg.baseline.node_selection == "min_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1])

                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

                return [i[0] for i in idx_score[max_width:]]

        return []

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, data["n_cons"], self.cfg.prob.n_objs, data["obj_coeffs"],
                     data["cons_coeffs"], data["rhs"])

    def set_var_layer(self, env):
        # print("Set var layer")
        env.set_var_layer(-1)
