import numpy as np

from scorer import NodeScorer


class MISHeuristicNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def preprocess_layer(self, layer):
        return [np.sum(node["s"]) for node in layer]
