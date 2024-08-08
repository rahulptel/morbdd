def trainer_factory(cfg):
    # Transformer
    if cfg.model.type == "tf":
        from .ann import TransformerTrainer
        return TransformerTrainer(cfg)

    # Graph Transformer
    elif cfg.model.type == "gtf":
        from .ann import TransformerTrainer
        return TransformerTrainer(cfg)
