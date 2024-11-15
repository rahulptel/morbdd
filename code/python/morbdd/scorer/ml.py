from .scorer import NodeScorer


class MLNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError


class XGBNodeScorer(MLNodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        pass

    def get_score(self, *args):
        pass


class NeuralNodeScorer(MLNodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self):
        pass

    def get_score(self, *args):
        pass
