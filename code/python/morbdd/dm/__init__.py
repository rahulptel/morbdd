def dm_factory(cfg):
    if cfg.prob.name == "knapsack":
        from .kp import KnapsackDataManager
        return KnapsackDataManager(cfg)

    elif cfg.prob.name == "indepset":
        from .ind import IndepsetDataManager
        return IndepsetDataManager(cfg)

    elif cfg.prob.name == "tsp":
        from .tsp import TSPDataManager
        return TSPDataManager(cfg)