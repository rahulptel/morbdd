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
from morbdd.utils import compute_dd_size
from morbdd.utils import compute_dd_width
from morbdd.utils import get_env
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
        self.orig_width = None

        self.restricted_dd = None
        self.restricted_size = None
        self.restricted_width = None

        self.reduced_dd = None
        self.reduced_size = None
        self.reduced_width = None

    def set_inst(self, *args):
        raise NotImplementedError

    def select_nodes(self, *args):
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
    def node_select_random(idx_score, max_width, layer_width):
        idxs = np.arange(layer_width)
        np.random.shuffle(idxs)

        return idxs[max_width:]

    def post_process(self, orig_dd, pid):
        self.orig_size = compute_dd_size(orig_dd)
        self.orig_width = compute_dd_width(orig_dd)
        true_pf = self.load_pf(pid)

        self.cardinality_raw, self.cardinality = -1, -1
        if true_pf is not None and self.pred_pf is not None:
            res = self.metric_calculator.compute_cardinality(true_pf, self.pred_pf)
            self.cardinality_raw = res['cardinality_raw']
            self.cardinality = res['cardinality']
            self.precision = res['precision']

    def save_result(self, seed, pid, build_time, pareto_time):
        total_time = build_time + pareto_time
        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.split, pid, total_time, self.orig_size, self.restricted_size,
                            self.reduced_size, self.orig_width, self.restricted_width, self.reduced_width,
                            self.cardinality, self.cardinality_raw, self.precision, len(self.pred_pf), build_time,
                            pareto_time]],
                          columns=["size", "split", "pid", "total_time", "orig_size", "rest_size", "reduced_size",
                                   "orig_width", "rest_width", "reduced_width", "cardinality", "cardinality_raw",
                                   "pred_precision", "n_pred_pf", "build_time", "pareto_time"])

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
            if len(layer) > max_width:
                removed_idx = self.select_nodes(rng, layer, max_width)
                if len(removed_idx):
                    env.approximate_layer(lid, CONST.RESTRICT, 1, removed_idx)

        # Generate terminal layer
        env.generate_next_layer()

    def reduce_dd(self, env):
        pass

    def run_pipeline(self, seed, pid, data, bdd, order, max_width):
        print(f"Pid: {pid}, seed: {seed}")
        signal.signal(signal.SIGALRM, handle_timeout)
        rng = random.Random(seed)
        env = get_env()
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
        self.restricted_dd = env.get_dd()
        self.restricted_size = compute_dd_size(self.restricted_dd)
        self.restricted_width = compute_dd_width(self.restricted_dd)

        start = time.time()
        self.reduce_dd(env)
        build_time += time.time() - start
        self.reduced_dd = env.get_dd()
        self.reduced_size = compute_dd_size(self.reduced_dd)
        self.reduced_width = compute_dd_width(self.reduced_dd)

        # Compute pareto frontier
        try:
            signal.alarm(1800)
            env.compute_pareto_frontier()
            self.pred_pf = env.get_frontier()["z"]
            pareto_time = env.get_time(CONST.TIME_PARETO)
        except:
            self.pred_pf = None
            pareto_time = 1800
        signal.alarm(0)

        self.post_process(bdd, pid)
        self.save_result(seed, pid, build_time, pareto_time)

    def worker(self, rank):
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
            file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive, file, format="json")
            if bdd is not None:
                width = compute_dd_width(bdd)
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
