import numpy as np


class NodeSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, *args):
        raise NotImplementedError


class WidthBasedNodeSelector(NodeSelector):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select(self, scores, width=None):
        if width is None:
            width = self.cfg.selector.width

        return np.arange(len(scores))[:width]


class ThresholdBasedNodeSelector(NodeSelector):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select(self, scores, threshold=None):
        if threshold is None:
            threshold = self.cfg.selector.threshold

        return [i for i, s in enumerate(scores) if s > threshold]
