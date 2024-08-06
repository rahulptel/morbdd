def trainer_factory(cfg):
    if cfg.prob.name == "kp":
        from kp import KnapsackTrainer
        return KnapsackTrainer(cfg)

    elif cfg.prob.name == "ind":
        from ind import IndepsetTrainer
        return IndepsetTrainer(cfg)
