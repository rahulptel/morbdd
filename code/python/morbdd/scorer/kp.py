from morbdd import ResourcePaths
from .heuristic import HeuristicNodeScorer

resource = ResourcePaths()


class KnapsackHeuristicNodeScorer(HeuristicNodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def preprocess_layer(self, layer):
        return [n["s"][0] for n in layer]
