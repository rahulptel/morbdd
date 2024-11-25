import json
import signal
import time

import numpy as np
import torch

from morbdd import ResourcePaths
from morbdd.deployer.deployer import Deployer
from morbdd.scorer import scorer_factory
from morbdd.utils import handle_timeout, compute_dd_size, compute_dd_width
from morbdd.utils.const import *
from morbdd.utils.tsp import TSPResult
from morbdd.utils.tsp import compute_stat_features
from morbdd.utils.tsp import get_env
from morbdd.utils.tsp import get_instance_data

resource = ResourcePaths()


class TSPDeployer(Deployer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.node_scorer = scorer_factory(cfg)
        self.exact_dd = json.load(
            open(resource.bdd / f"tsp/{cfg.prob.size}/tsp_dd.json", "r")
        )
        self.layer_sizes = [len(layer) for layer in self.exact_dd]
        self.rest_width = int(max(self.layer_sizes) * (cfg.rest_width / 100))
        print("Size of the exact dd: ", sum(self.layer_sizes))
        print("Width of the exact dd: ", max(self.layer_sizes))
        print("Max width of the restricted dd: ", self.rest_width)

    def build_dd(self, env, inst):
        raise NotImplementedError

    def run_pipeline(self, inst, true_pf=None):
        result = TSPResult()

        signal.signal(signal.SIGALRM, handle_timeout)
        env = get_env(n_objs=self.cfg.prob.n_objs)
        env.reset()
        env.set_inst(
            self.cfg.prob.n_vars,
            self.cfg.prob.n_objs,
            inst["dists"].astype(int).tolist(),
        )
        env.initialize_dd_constructor()
        result.build_time = time.time()
        self.build_dd(env, inst)
        result.build_time = time.time() - result.build_time

        # Compute pareto frontier
        try:
            signal.alarm(1800)
            result.pareto_time = time.time()
            env.compute_pareto_frontier()
            result.pareto_time = time.time() - result.pareto_time
            frontier = env.get_frontier()
            result.pred_pf, result.pred_sol = frontier["z"], frontier["x"]
        except TimeoutError:
            result.pred_pf, result.pred_sol = None, None
            result.pareto_time = 1800
        signal.alarm(0)
        result.total_time = result.build_time + result.pareto_time

        if result.pred_pf is not None:
            restricted_dd = env.get_dd()
            result.restricted_size = compute_dd_size(restricted_dd)
            result.restricted_width = compute_dd_width(restricted_dd)

        if true_pf is not None:
            res = self.metric_calculator.compute_cardinality(true_pf, result.pred_pf)
            result.cardinality_raw = res["cardinality_raw"]
            result.cardinality = res["cardinality"]
            result.precision = res["precision"]

        return result

    def deploy(self):
        for pid in range(self.cfg.from_pid, self.cfg.to_pid):
            # Load instance
            inst = get_instance_data(self.cfg.prob.size, self.cfg.split, pid, seed=7)
            sol_path = (
                resource.sol
                / "tsp"
                / self.cfg.prob.size
                / self.cfg.split
                / f"sol_{pid}.npz"
            )
            if not sol_path.exists():
                continue

            true_pf = np.load(sol_path)
            true_pf = true_pf["z"]
            result = self.run_pipeline(inst, true_pf=true_pf)
            self.save_result(pid, result, self.rest_width)


class TSPHeuristicDeployer(TSPDeployer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_dd(self, env, inst):
        self.node_scorer.set_inst(inst)

        lid = 2
        while True:
            # print("Building layer: ", lid)
            is_done = env.generate_next_layer()
            layer = env.get_layer(lid)
            # print("Size: ", len(layer))
            if len(layer) > self.rest_width:
                # print("\tRestricting")
                # Sort nodes in ascending order of scores and remove the last ones
                scores = self.node_scorer.get_score(layer)
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                idx_scores.sort(key=lambda x: x[1])
                nodes_to_remove = [i for i, _ in idx_scores[self.rest_width :]]
                env.approximate_layer(lid, RESTRICT, nodes_to_remove)
                # print("\tSize: ", len(env.get_layer(lid)))
            lid += 1
            if is_done:
                break


class TSPGTDeployer(TSPDeployer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_dd(self, env, inst):
        coords = (torch.from_numpy(inst["coords"]) / GRID_DIM).float().unsqueeze(0)
        dists = (
            (torch.from_numpy(inst["dists"]) / MAX_DIST_ON_GRID).float().unsqueeze(0)
        )
        node_feat = torch.cat((coords, compute_stat_features(dists)), dim=-1)
        node_emb = self.node_scorer.get_node_emb(node_feat, dists)

        # Restrict and build
        lid = 2
        while True:
            # print("Building layer: ", lid)
            is_done = env.generate_next_layer()
            layer = env.get_layer(lid)
            # print("Size: ", len(layer))
            if len(layer) > self.rest_width:
                # print("\tRestricting")
                scores = self.node_scorer.get_score(
                    lid - 1, node_emb, layer, n_vars=self.cfg.prob.n_vars
                )
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                # Sort in descending order of scores
                idx_scores.sort(key=lambda x: x[1], reverse=True)
                nodes_to_remove = [i for i, _ in idx_scores[self.rest_width :]]
                env.approximate_layer(lid, RESTRICT, nodes_to_remove)

            lid += 1
            if is_done:
                break
