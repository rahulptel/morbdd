from morbdd.utils.const import *


def trainer_factory(cfg):
    if cfg.prob.prefix == KNAPSACK:
        if cfg.model.type == GRADIENT_BOOSTED_TREE:
            from .kp import XGBTrainer

            return XGBTrainer(cfg)

    if cfg.prob.prefix == TSP:
        if cfg.model.type == GRAPH_TRANSFORMER:
            from .tsp import GTTrainer

            return GTTrainer(cfg)
