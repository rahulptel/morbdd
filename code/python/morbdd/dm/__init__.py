def dm_factory(cfg):
    if cfg.prob.name == "knapsack":
        from .kp import KnapsackDataManager
        return KnapsackDataManager(cfg)

    elif cfg.prob.name == "mis":
        from .mis import MISDataManager
        return MISDataManager(cfg)
