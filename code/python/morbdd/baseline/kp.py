import numpy as np
from pymoo.core.problem import Problem

from .nsga2 import EABaseline
from .wrbdd import WidthRestrictedBDD


class MultiObjectiveKnapsack(Problem):
    def __init__(self, inst_data):
        super().__init__(n_var=inst_data["n_var"], n_obj=inst_data["n_obj"], n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.W = inst_data["weight"]
        self.V = inst_data["value"]
        self.C = inst_data["capacity"]

    def _evaluate(self, x, out, *args, **kwargs):
        f = [-np.sum(v * x, axis=1) for v in self.V]
        out["F"] = np.column_stack(f)
        out["G"] = (np.sum(self.W * x, axis=1) - self.C)


class KnapsackEABaseline(EABaseline):
    def __init__(self, cfg):
        super(EABaseline, self).__init__(cfg)

    @staticmethod
    def min_converter(z):
        return -np.array(z)

    def set_problem(self):
        self.problem = MultiObjectiveKnapsack(self.inst_data)


class KnapsackWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, 1, self.cfg.prob.n_objs, list(np.array(data["value"]).T),
                     [data["weight"]], [data["capacity"]])

    def select_nodes(self, rng, layer, max_width):
        # print(f"Max width is {max_width}, layer width is {len(layer)}")
        n_layer = len(layer)
        if max_width < n_layer:
            # print("Restricting...")
            if self.cfg.baseline.node_selection == "random":
                idxs = np.arange(n_layer)
                rng.shuffle(idxs)

                return idxs[:max_width]
            elif self.cfg.baseline.node_selection == "min_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1])

                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_weight":
                idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

                return [i[0] for i in idx_score[max_width:]]

        return []

    def reduce_dd(self, env):
        print("Reducing dd...")
        env.reduce_dd()
