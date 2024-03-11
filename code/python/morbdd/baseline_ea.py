import json

import hydra
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from laser import resource_path
from laser.utils import get_instance_data


class MultiObjectiveKnapsack(Problem):
    def __init__(self, n_obj, n_var, W, P, C):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        f = [-np.sum(p * x, axis=1) for p in self.P]

        out["F"] = np.column_stack(f)
        out["G"] = (np.sum(self.W * x, axis=1) - self.C)


def save_result(cfg, res):
    result_path = resource_path / f"ea/{cfg.deploy.algorithm}/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}"
    result_path.mkdir(exist_ok=True, parents=True)
    np.savez(result_path / f"{cfg.deploy.pid}.npz", X=res.X, F=res.F)


@hydra.main(version_base='1.2', config_path='./configs', config_name='baseline_ea.yaml')
def main(cfg):
    inst_data = get_instance_data(cfg.prob.name, cfg.prob.size, cfg.deploy.split, cfg.deploy.pid)
    problem = MultiObjectiveKnapsack(cfg.prob.num_objs,
                                     cfg.prob.num_vars,
                                     inst_data["weight"],
                                     inst_data["value"],
                                     inst_data["capacity"])

    algorithm = None
    if cfg.deploy.algorithm == "nsga2":
        algorithm = NSGA2(pop_size=1000,
                          sampling=BinaryRandomSampling(),
                          crossover=TwoPointCrossover(),
                          mutation=BitflipMutation(),
                          eliminate_duplicates=True)
    assert algorithm is not None

    # print(cfg.deploy.time_limit)
    termination = get_termination("time", cfg.deploy.time_limit)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)

    save_result(cfg, res)


if __name__ == '__main__':
    main()
