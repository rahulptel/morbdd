def trainer_factory(cfg):
    # Transformer
    if cfg.model.type == "tf":
        from .ann import TransformerTrainer
        return TransformerTrainer(cfg)

    # Graph Transformer
    elif cfg.model.type == "gtf":
        from .ann import TransformerTrainer
        return TransformerTrainer(cfg)

    elif cfg.model.type == "gbt":
        from .xgb import XGBTrainer
        return XGBTrainer(cfg)

    elif cfg.model.type == "gbt_rank":
        from .xgb import XGBRankTrainer
        return XGBRankTrainer(cfg)
