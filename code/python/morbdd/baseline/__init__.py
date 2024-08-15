def baseline_factory(cfg):
    if cfg.baseline.type == "nsga2":
        if cfg.prob.name == "knapsack":
            from .nsga2 import KnapsackEABaseline
            return KnapsackEABaseline(cfg)
        elif cfg.prob.name == "indepset":
            from .nsga2 import IndepsetEABaseline
            return IndepsetEABaseline(cfg)
    elif cfg.baseline.type == "wrbdd":
        if cfg.prob.name == "knapsack":
            from .wrbdd import KnapsackWidthRestrictedBDD
            return KnapsackWidthRestrictedBDD(cfg)
        elif cfg.prob.name == "indepset":
            from .wrbdd import IndepsetWidthRestrictedBDD
            return IndepsetWidthRestrictedBDD(cfg)
