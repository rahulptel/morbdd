import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

from morbdd import ResourcePaths
from .scorer import NodeScorer

resource = ResourcePaths()


class XGBNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        exp_path = (
            resource.pretrained
            / self.cfg.prob.prefix
            / self.cfg.prob.size
            / self.cfg.model.type
        )
        if not exp_path.exists():
            raise FileNotFoundError()

        exp_cfg = OmegaConf.load(exp_path / "config.yaml")
        model_path = exp_path / f"best_model.json"
        print("Loading model: ", model_path, ", Exists: ", model_path.exists())
        if not model_path.exists():
            raise FileNotFoundError()

        self.model = xgb.Booster(**exp_cfg.model)
        self.model.load_model(model_path)

    def get_score(self, features):
        return self.model.predict(
            xgb.DMatrix(np.array(features)),
            iteration_range=(0, self.model.best_iteration + 1),
        )
