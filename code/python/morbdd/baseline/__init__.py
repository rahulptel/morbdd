def baseline_factory(cfg):
    if cfg.baseline == "nsga2":
        if cfg.prob == "kp":
            from .nsga2 import KnapsackEABaseline
            return KnapsackEABaseline(cfg)
        elif cfg.prob == "ind":
            from .nsga2 import IndepsetEABaseline
            return IndepsetEABaseline(cfg)
    elif cfg.baseline == "rwbdd":
        if cfg.prob == "kp":
            from .wrbdd import KnapsackWidthRestrictedBDD
            return KnapsackWidthRestrictedBDD(cfg)
        elif cfg.prob == "ind":
            from .wrbdd import IndepsetWidthRestrictedBDD
            return IndepsetWidthRestrictedBDD(cfg)
