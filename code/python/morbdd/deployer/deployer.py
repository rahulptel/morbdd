from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from morbdd import ResourcePaths
from morbdd.utils import MetricCalculator

resource = ResourcePaths()


class Deployer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric_calculator = MetricCalculator(self.cfg.prob.n_objs)

    def save_result(self, pid, result, max_width, seed=7):
        df = pd.DataFrame(
            [
                [
                    self.cfg.prob.size,
                    self.cfg.split,
                    pid,
                    result.total_time,
                    result.orig_size,
                    result.restricted_size,
                    result.reduced_size,
                    result.orig_width,
                    result.restricted_width,
                    result.reduced_width,
                    result.cardinality,
                    result.cardinality_raw,
                    result.precision,
                    len(result.pred_pf),
                    result.build_time,
                    result.pareto_time,
                ]
            ],
            columns=[
                "size",
                "split",
                "pid",
                "total_time",
                "orig_size",
                "rest_size",
                "reduced_size",
                "orig_width",
                "rest_width",
                "reduced_width",
                "cardinality",
                "cardinality_raw",
                "pred_precision",
                "n_pred_pf",
                "build_time",
                "pareto_time",
            ],
        )

        save_path = (
            resource.restricted_sol
            / self.cfg.prob.name
            / self.cfg.prob.size
            / self.cfg.split
            / self.cfg.scorer
            / str(max_width)
        )
        # save_path = save_path / f"{exp_path}" / str(cfg.baseline.max_width)
        save_path.mkdir(parents=True, exist_ok=True)
        result_path = save_path / f"{pid}_{seed}.csv"
        print(df)
        df.to_csv(result_path)

        # Save restricted sols
        np.savez(
            save_path / f"rest_sol_{pid}_{seed}.npz",
            x=np.array(result.pred_sol),
            z=np.array(result.pred_pf),
        )

    @abstractmethod
    def deploy(self):
        pass
