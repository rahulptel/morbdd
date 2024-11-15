from .wrbdd import WidthRestrictedBDD
import numpy as np
from .nsga2 import EABaseline
from pymoo.core.problem import Problem


class MultiObjectiveIndepset(Problem):
    def __init__(self, inst_data):
        super().__init__(n_var=inst_data["n_var"], n_obj=inst_data["n_obj"], n_ieq_constr=inst_data["n_cons"], xl=0,
                         xu=1, vtype=bool)
        self.W = inst_data["adj_mat"]
        self.V = inst_data["obj_coeffs"]

    def _evaluate(self, x, out, *args, **kwargs):
        f = [-np.sum(v * x) for v in self.V]
        out["F"] = np.column_stack(f)
        g = [np.sum(w * x) - 1 for w in self.W]
        out["G"] = np.column_stack(g)


class IndepsetEABaseline(EABaseline):
    def __init__(self, cfg):
        EABaseline.__init__(self, cfg)

    @staticmethod
    def min_converter(z):
        return -np.array(z)

    def set_problem(self):
        self.problem = MultiObjectiveIndepset(self.inst_data)


class IndepsetWidthRestrictedBDD(WidthRestrictedBDD):
    def __init__(self, cfg):
        super().__init__(cfg)

    def select_nodes(self, rng, layer, max_width):
        n_layer = len(layer)
        if max_width < n_layer:
            if self.cfg.baseline.node_selection == "random":
                idxs = np.arange(n_layer)
                rng.shuffle(idxs)

                return idxs[:max_width]
            if self.cfg.baseline.node_selection == "min_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1])

                return [i[0] for i in idx_score[max_width:]]
            elif self.cfg.baseline.node_selection == "max_items":
                idx_score = [(i, np.sum(n["s"])) for i, n in enumerate(layer)]
                idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

                return [i[0] for i in idx_score[max_width:]]

        return []

    def set_inst(self, env, data):
        env.set_inst(self.cfg.prob.n_vars, data["n_cons"], self.cfg.prob.n_objs, data["obj_coeffs"],
                     data["cons_coeffs"], data["rhs"])

    def set_var_layer(self, env):
        # print("Set var layer")
        env.set_var_layer(-1)
