import json
import signal
import time

import hydra
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from morbdd import ResourcePaths as path
from morbdd.utils import MetricCalculator
from morbdd.utils import handle_timeout
from morbdd.utils.tsp import Result
from morbdd.utils.tsp import get_instance_data, get_env

RESTRICT = 1


class TSPNodeScorer:
    def __init__(self, edge_agg, next_node):
        self.inst = None
        self.edge_agg = edge_agg
        self.next_node = next_node
        self.edge_rank = None

    def reset_inst(self, inst):
        self.inst = inst
        self.rank_edges()

    def rank_edges(self):
        self.edge_rank = []
        for obj in self.inst["dists"]:
            obj_flatten = obj.flatten()
            obj_flatten = rankdata(obj_flatten, method="dense")
            obj = obj_flatten.reshape(obj.shape)
            self.edge_rank.append(obj)
        self.edge_rank = np.array(self.edge_rank)

        if self.edge_agg == "mean":
            self.edge_rank = np.mean(self.edge_rank, axis=0)
        elif self.edge_agg == "max":
            self.edge_rank = np.max(self.edge_rank, axis=0)
        elif self.edge_agg == "min":
            self.edge_rank = np.min(self.edge_rank, axis=0)
        else:
            raise ValueError("Method must be either 'mean' or 'max'")

    def score_nodes(self, layer):
        scores = []
        for nid, node in enumerate(layer):
            last_visit = node[-1]
            to_visit = [i for i, n in enumerate(node[:-1]) if n == 0]
            if len(to_visit):
                if self.next_node == "min":
                    score = np.min([self.edge_rank[last_visit][tv] for tv in to_visit])
                elif self.next_node == "max":
                    score = np.max([self.edge_rank[last_visit][tv] for tv in to_visit])
                else:
                    raise ValueError(
                        "Node selection must be either 'greedy' or 'robust'"
                    )
                scores.append(score)
            else:
                # Static score for all nodes as we cannot discriminate between them
                scores.append(1)

        return scores


def compare_dd():
    pass


def compare_pf():
    pass


def save_result(cfg, result, seed, pid):
    df = pd.DataFrame(
        [
            [
                cfg.prob.size,
                cfg.split,
                pid,
                result.total_time,
                result.orig_size,
                result.restricted_size,
                result.reduced_size,
                result.orig_width,
                result.restricted_width,
                result.reduced_width,
                result.cardinality,
                result.cardinality_raw,
                result.precision,
                len(result.pred_pf),
                result.build_time,
                result.pareto_time,
            ]
        ],
        columns=[
            "size",
            "split",
            "pid",
            "total_time",
            "orig_size",
            "rest_size",
            "reduced_size",
            "orig_width",
            "rest_width",
            "reduced_width",
            "cardinality",
            "cardinality_raw",
            "pred_precision",
            "n_pred_pf",
            "build_time",
            "pareto_time",
        ],
    )

    pid = str(pid) + f"_{seed}.csv"
    save_path = (
        path.resource / "restricted_sols" / cfg.prob.name / cfg.prob.size / cfg.split
    )
    save_path = (
        save_path
        / f"{cfg.baseline.edge_agg}_{cfg.baseline.next_node}"
        / str(cfg.baseline.max_width)
    )
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / pid
    print(df)
    df.to_csv(save_path)


def build_dd(env, max_width, scorer):
    # Restrict and build
    lid = 2
    while True:
        print("Building layer: ", lid)
        is_done = env.generate_next_layer()
        layer = env.get_layer(lid)
        print("Size: ", len(layer))
        if len(layer) > max_width:
            print("\tRestricting")
            # Sort nodes in ascending order of scores and remove the last ones
            scores = scorer.score_nodes(layer)
            idx_scores = [(i, s) for i, s in enumerate(scores)]
            idx_scores.sort(key=lambda x: x[1])
            nodes_to_remove = [i for i, _ in idx_scores[max_width:]]
            # nodes_to_remove.sort()
            # print(nodes_to_remove)
            env.approximate_layer(lid, RESTRICT, nodes_to_remove)
            print("\tSize: ", len(env.get_layer(lid)))
        lid += 1
        if is_done:
            break


def run_pipeline(
    cfg, inst, max_width, scorer, metric_calculator, result, exact_dd=None, true_pf=None
):
    signal.signal(signal.SIGALRM, handle_timeout)
    env = get_env()

    env.reset()
    env.set_inst(
        cfg.prob.n_vars,
        cfg.prob.n_objs,
        inst["dists"].astype(int).tolist(),
    )
    env.initialize_dd_constructor()
    start = time.time()
    build_dd(env, max_width, scorer)
    result.build_time = time.time() - start

    # Compute pareto frontier
    try:
        signal.alarm(1800)
        start = time.time()
        env.compute_pareto_frontier()
        result.pareto_time = time.time() - start
        result.pred_pf = env.get_frontier()["z"]
    except:
        result.pred_pf = None
        result.pareto_time = 1800
    signal.alarm(0)
    result.total_time = result.build_time + result.pareto_time
    if result.pred_pf is not None:
        restricted_dd = env.get_dd()
        layer_sizes = [len(layer) for layer in restricted_dd]
        result.restricted_size = sum(layer_sizes)
        result.restricted_width = max(layer_sizes)

    if true_pf is not None:
        res = metric_calculator.compute_cardinality(true_pf, result.pred_pf)
        result.cardinality_raw = res["cardinality_raw"]
        result.cardinality = res["cardinality"]
        result.precision = res["precision"]

    return result


@hydra.main(
    config_path="../configs", config_name="06_baseline_tsp.yaml", version_base="1.2"
)
def main(cfg):
    exact_dd = json.load(
        open(path.bdd / f"{cfg.prob.name}/{cfg.prob.size}/tsp_dd.json", "r")
    )
    layer_sizes = [len(layer) for layer in exact_dd]
    max_width = int(max(layer_sizes) * (cfg.baseline.max_width / 100))
    print("Size of the exact dd: ", sum(layer_sizes))
    print("Width of the exact dd: ", max(layer_sizes))
    print("Max width of the restricted dd: ", max_width)

    scorer = TSPNodeScorer(cfg.baseline.edge_agg, cfg.baseline.next_node)
    metric_calculator = MetricCalculator(cfg.prob.n_vars)
    for pid in range(cfg.from_pid, cfg.to_pid):
        result = Result()
        result.orig_size = sum(layer_sizes)
        result.orig_width = max(layer_sizes)
        result.edge_agg = cfg.baseline.edge_agg
        result.next_node = cfg.baseline.next_node

        # Load instance
        inst = get_instance_data(cfg.prob.size, cfg.split, pid, seed=7)
        # Load true PF
        true_pf = None
        sol_path = (
            path.sol / cfg.prob.name / cfg.prob.size / cfg.split / f"sol_{pid}.npz"
        )
        if sol_path.exists():
            true_pf = np.load(sol_path)
            true_pf = true_pf["z"]

        scorer.reset_inst(inst)
        # Build dd layer-by-layer with pruning using node information
        result = run_pipeline(
            cfg,
            inst,
            max_width,
            scorer,
            metric_calculator,
            result,
            exact_dd=exact_dd,
            true_pf=true_pf,
        )

        save_result(cfg, result, 7, pid)


if __name__ == "__main__":
    main()
