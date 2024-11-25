import signal
import time

import numpy as np
import xgboost as xgb

from morbdd import ResourcePaths as path
from morbdd.utils import compute_dd_width, compute_dd_size, Result
from morbdd.utils import get_env
from morbdd.utils import get_instance_data
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
from morbdd.utils import read_from_zip
from morbdd.utils.const import TIME_PARETO, RESTRICT
from morbdd.utils.kp import LayerToFeatureConverter
from .deployer import Deployer
from ..scorer import scorer_factory


class KnapsackDeployer(Deployer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.node_scorer = scorer_factory(cfg)
        self.exact_dd = None
        self.layer_sizes = None
        self.rest_width = None

    def build_dd(self, *args):
        raise NotImplementedError

    def run_pipeline(self, inst_data, order, true_pf=None):
        result = Result()

        env = get_env(n_objs=self.cfg.prob.n_vars)
        signal.signal(signal.SIGALRM, handle_timeout)
        env.reset(
            self.cfg.prob.problem_type,
            self.cfg.prob.preprocess,
            self.cfg.prob.pf_enum_method,
            self.cfg.prob.maximization,
            self.cfg.prob.dominance,
            self.cfg.prob.bdd_type,
            self.cfg.prob.maxwidth,
            order,
        )

        env.set_inst(
            inst_data["n_vars"],
            inst_data["n_cons"],
            inst_data["n_objs"],
            list(np.array(inst_data["value"]).T),
            [inst_data["weight"]],
            [inst_data["capacity"]],
        )
        env.preprocess_inst()
        env.initialize_dd_constructor()

        # Build restricted DD
        result.build_time = time.time()
        self.build_dd(env, inst_data, order)
        result.build_time = time.time() - result.build_time
        # Fetch DD and compute stats
        rest_dd = env.get_dd()
        result.restricted_size = compute_dd_size(rest_dd)
        result.restricted_width = compute_dd_width(rest_dd)

        # Reduce DD
        start = time.time()
        env.reduce_dd()
        result.build_time += time.time() - start
        rest_dd = env.get_dd()
        result.reduced_width = compute_dd_width(rest_dd)
        result.reduced_size = compute_dd_size(rest_dd)

        # print(f"/7/10: Computing Pareto Frontier...")
        try:
            signal.alarm(1800)
            env.compute_pareto_frontier()
            frontier = env.get_frontier()
            result.pareto_time = env.get_time(TIME_PARETO)
            result.pred_pf, result.pred_sol = frontier["z"], frontier["x"]
        except TimeoutError:
            is_pf_computed = False
            result.pred_pf, result.pred_sol = None, None
            result.pareto_time = 1800
            # print(f"PF not computed within {self.cfg.prob.time_limit} for pid {pid}")
        # print(f"PF computed successfully for pid {pid}")
        signal.alarm(0)
        result.total_time = result.build_time + result.pareto_time

        if result.pred_pf is not None:
            reduced_dd = env.get_dd()
            result.reduced_size = compute_dd_size(reduced_dd)
            result.reduced_width = compute_dd_width(reduced_dd)

        if true_pf is not None:
            res = self.metric_calculator.compute_cardinality(true_pf, result.pred_pf)
            result.cardinality_raw = res["cardinality_raw"]
            result.cardinality = res["cardinality"]
            result.precision = res["precision"]

        return result

    def deploy(self):
        for pid in range(self.cfg.from_pid, self.cfg.to_pid):
            archive = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
            file = f"{self.cfg.prob.size}/{self.cfg.deploy.split}/{pid}.json"
            self.exact_dd = read_from_zip(archive, file, format="json")
            if self.exact_dd is not None:
                self.layer_sizes = [len(layer) for layer in self.exact_dd]
                self.rest_width = int(
                    max(self.layer_sizes) * (self.cfg.rest_width / 100)
                )

                # Read instance
                inst_data = get_instance_data(
                    self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid
                )
                order = get_static_order(
                    self.cfg.prob.name, self.cfg.deploy.order_type, inst_data
                )
                result = self.run_pipeline(inst_data, order)
                self.save_result(pid, result, self.rest_width)


class KnapsackHeuristicDeployer(KnapsackDeployer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_dd(self, env, inst_data, order):
        lid = 0
        # Restrict and build
        while lid < self.cfg.prob.n_vars - 1:
            env.generate_next_layer()
            lid += 1
            layer = env.get_layer(lid)
            if len(layer) > self.rest_width:
                scores = self.node_scorer.get_score(layer)
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                idx_scores.sort(key=lambda x: x[1], reverse=self.cfg.reverse)
                nodes_to_remove = [i for i, _ in idx_scores[self.rest_width :]]
                env.approximate_layer(lid, RESTRICT, 1, nodes_to_remove)

        # Generate terminal layer
        env.generate_next_layer()


class KnapsackGBTDeployer(KnapsackDeployer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.converter = LayerToFeatureConverter()

    def build_dd(self, env, inst_data, order):
        self.converter.reset(inst_data, order)

        lid = 0
        while lid < self.cfg.prob.n_vars - 1:
            env.generate_next_layer()
            lid += 1
            layer = env.get_layer(lid)
            if len(layer) > self.rest_width:
                # print("Restricting...")
                features = self.converter.convert(lid, layer)
                scores = self.node_scorer.get_score(xgb.DMatrix(np.array(features)))
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                idx_scores = sorted(
                    idx_scores, key=lambda x: (x[1], -x[0]), reverse=True
                )
                nodes_to_remove = [i for i, _ in idx_scores[self.rest_width :]]
                env.approximate_layer(lid, RESTRICT, 1, nodes_to_remove)

        # Generate terminal layer
        env.generate_next_layer()
