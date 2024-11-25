from morbdd.deployer.kp import KnapsackHeuristicDeployer
from morbdd.utils.const import (
    KNAPSACK,
    MIS,
    TSP,
    GRADIENT_BOOSTED_TREE,
    HEURISTIC_STATE_SCORER,
    GRAPH_TRANSFORMER,
)


def deployer_factory(cfg):
    if cfg.prob.prefix == KNAPSACK:
        if cfg.scorer.type == GRADIENT_BOOSTED_TREE:
            from .kp import KnapsackGBTDeployer

            return KnapsackGBTDeployer(cfg)
        elif cfg.scorer.type == HEURISTIC_STATE_SCORER:
            from .kp import KnapsackHeuristicDeployer

            return KnapsackHeuristicDeployer(cfg)

    elif cfg.prob.prefix == MIS:
        if cfg.scorer.type == HEURISTIC_STATE_SCORER:
            from .ind import IndepsetDeployer

            return IndepsetHeuristicDeployer(cfg)

    elif cfg.prob.prefix == TSP:
        if cfg.scorer.type == GRAPH_TRANSFORMER:
            from .tsp import TSPGTDeployer

            return TSPGTDeployer(cfg)
        elif cfg.scorer.type == HEURISTIC_STATE_SCORER:
            from .tsp import TSPHeuristicDeployer

            return TSPHeuristicDeployer(cfg)
