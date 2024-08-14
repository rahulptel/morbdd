import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from .baseline import Baseline
from morbdd.utils import read_from_zip


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


class EABaseline(Baseline):
    def __init__(self, cfg):
        Baseline.__init__(self, cfg)
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.termination = None
        self.algorithm = None
        self.X, self.F = [], []

        self.set_sampling()
        self.set_crossover()
        self.set_mutation()
        self.set_termination()
        self.set_algorithm()

        self.res = None

    def reset(self):
        self.hv_approx = []
        self.cardinality = []
        self.precision = []
        self.times = []
        self.problem = None
        self.inst_data = None

        self.X, self.F = [], []
        self.res = None

    def set_sampling(self):
        if self.cfg.baseline.sampling == "BinaryRandomSampling":
            self.sampling = BinaryRandomSampling()

        assert self.sampling is not None

    def set_crossover(self):
        if self.cfg.baseline.crossover == "TwoPointCrossover":
            self.crossover = TwoPointCrossover()

        assert self.crossover is not None

    def set_mutation(self):
        if self.cfg.baseline.mutation == "BitflipMutation":
            self.mutation = BitflipMutation()

        assert self.mutation is not None

    def set_termination(self):
        self.termination = get_termination("time", self.cfg.time_limit)

        assert self.termination is not None

    def set_algorithm(self):
        if self.cfg.baseline == "nsga2":
            self.algorithm = NSGA2(pop_size=self.cfg.baseline.population_size,
                                   sampling=self.sampling,
                                   crossover=self.crossover,
                                   mutation=self.mutation,
                                   eliminate_duplicates=self.cfg.baseline.eliminate_duplicates)

        assert self.algorithm is not None

    @staticmethod
    def min_converter(z):
        raise NotImplementedError

    def save_intermediate_result(self):
        self.X.append(self.res.X)
        self.F.append(self.res.F)

    def save_final_result(self, pid):
        archive = path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"
        file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
        sol = read_from_zip(archive, file, format="json")
        # Ignore instances not solved within time limit
        if sol is None:
            return
        z = np.array(sol["z"])
        z = self.min_converter(z)
        norm = np.abs(np.min(z, axis=0))

        for pred_z in self.F:
            z_pred = self.min_converter(pred_z)
            z_pred_norm = z_pred / norm

            res = self.metric_calculator.compute_cardinality(z, z_pred)
            self.cardinality.append(res['cardinality'])
            self.precision.append(res['precision'])
            hv_res = [self.metric_calculator.compute_approx_hv(s, z_pred_norm)
                      for s in self.cfg.baseline.hv_approx_seeds]
            hv_res = np.mean([r['hv_approx'] for r in hv_res])
            self.hv_approx.append(hv_res)

        result_path = path.resource / f"ea/{self.cfg.baseline}/{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        result_path.mkdir(exist_ok=True, parents=True)
        np.savez(result_path / f"{pid}_metrics.npz", cardinality=np.mean(self.cardinality),
                 precision=np.mean(self.precision), hv_approx=np.mean(self.hv_approx))

    def worker(self, pid):
        self.inst_data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.deploy.split, pid)
        self.set_problem()
        for seed in self.cfg.baseline.seeds:
            self.res = minimize(self.problem, self.algorithm, self.termination, seed=seed, verbose=True)
            self.save_intermediate_result()
        self.save_final_result(pid)
        self.reset()

    def set_problem(self):
        raise NotImplementedError


class KnapsackEABaseline(EABaseline):
    def __init__(self, cfg):
        super(EABaseline, self).__init__(cfg)

    @staticmethod
    def min_converter(z):
        return -np.array(z)

    def set_problem(self):
        self.problem = MultiObjectiveKnapsack(self.inst_data)


class IndepsetEABaseline(EABaseline):
    def __init__(self, cfg):
        EABaseline.__init__(self, cfg)

    @staticmethod
    def min_converter(z):
        return -np.array(z)

    def set_problem(self):
        self.problem = MultiObjectiveIndepset(self.inst_data)
