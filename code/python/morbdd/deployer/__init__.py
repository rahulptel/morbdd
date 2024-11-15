def deployer_factory(cfg):
    if cfg.prob.name == "knapsack":
        if cfg.model.type == "gbt_rank":
            from .kp import KnapsackRankDeployer
            return KnapsackRankDeployer(cfg)
        elif cfg.model.type == "gbt":
            from .kp import KnapsackDeployer
            return KnapsackDeployer(cfg)
    elif cfg.prob.name == "indepset":
        from .ind import IndepsetDeployer
        return IndepsetDeployer(cfg)
