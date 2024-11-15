from .scorer import NodeScorer
import numpy as np


class ScalerStateNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_score(self, node_data):
        idx_score = [(i, n["s"][0]) for i, n in enumerate(node_data)]
        idx_score = sorted(idx_score,
                           key=lambda x: x[1],
                           reverse=True if self.cfg.strategy == "descending" else False)

        return idx_score


class SetStateNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_score(self, node_data):
        idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(node_data)]
        idx_score = sorted(idx_score,
                           key=lambda x: x[1],
                           reverse=True if self.cfg.strategy == "descending" else False)

        return idx_score
