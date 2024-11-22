from .scorer import NodeScorer
import numpy as np


class HeuristicNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_score(self, layer):
        layer = self.preprocess_layer(layer)
        idx_score = [(i, state_score) for i, state_score in enumerate(layer)]
        idx_score = sorted(
            idx_score,
            key=lambda x: x[1],
            reverse=True if self.cfg.strategy == "descending" else False,
        )

        return idx_score

    def preprocess_layer(self, layer):
        raise NotImplementedError
