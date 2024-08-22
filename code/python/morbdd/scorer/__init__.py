import sys


def scorer_factory(cfg):
    if cfg.scorer.type == "heuristic_scaler_state":
        from .heuristic import ScalerStateNodeScorer
        return ScalerStateNodeScorer(cfg)

    elif cfg.scorer.type == "heuristic_set_state":
        from .heuristic import SetStateNodeScorer
        return SetStateNodeScorer(cfg)

    elif cfg.scorer.type == "xgb":
        from .ml import XGBNodeScorer
        return XGBNodeScorer(cfg)

    elif cfg.scorer.type == "neural":
        from .ml import NeuralNodeScorer
        return NeuralNodeScorer(cfg)

    else:
        print("Invalid node scorer!")
        sys.exit()
