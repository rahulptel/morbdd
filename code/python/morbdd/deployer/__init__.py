def deployer_factory(cfg):
    if cfg.prob.name == "knapsack":
        from .kp import KnapsackDeployer
        return KnapsackDeployer(cfg)
    elif cfg.prob.name == "indepset":
        from .ind import IndepsetDeployer
        return IndepsetDeployer(cfg)
