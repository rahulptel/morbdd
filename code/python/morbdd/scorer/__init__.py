from morbdd.scorer.tsp import TSPHeuristicNodeScorer
from morbdd.utils.const import *


def scorer_factory(cfg):
    if cfg.prob.prefix == KNAPSACK:
        if cfg.scorer.name == HEURISTIC_STATE_SCORER:
            from .kp import KnapsackHeuristicNodeScorer

            return KnapsackHeuristicNodeScorer(cfg)
        if cfg.scorer.name == GRADIENT_BOOSTED_TREE:
            from .ml import XGBNodeScorer

            return XGBNodeScorer(cfg)
        raise ValueError("Unknown scorer {}".format(cfg.scorer.name))
    elif cfg.prob.prefix == MIS:
        if cfg.scorer.name == HEURISTIC_STATE_SCORER:
            from .mis import MISHeuristicNodeScorer

            return MISHeuristicNodeScorer(cfg)
        raise ValueError("Unknown scorer {}".format(cfg.scorer.name))
    elif cfg.prob.prefix == TSP:
        if cfg.scorer.name == HEURISTIC_STATE_SCORER:
            from .tsp import TSPHeuristicNodeScorer

            return TSPHeuristicNodeScorer(cfg)
        if cfg.scorer.name == GRAPH_TRANSFORMER:
            from .tsp import GTNodeScorer

            return GTNodeScorer(cfg)
        raise ValueError("Unknown scorer {}".format(cfg.scorer.name))
    else:
        raise ValueError("Unknown problem {}".format(cfg.prob.prefix))
