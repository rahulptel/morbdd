import json
import multiprocessing as mp
import signal
import time
from abc import ABC, abstractmethod

import pandas as pd

from morbdd import CONST
from morbdd import ResourcePaths as path
from morbdd.utils import get_env
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
import numpy as np


class DataManager(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def generate_instances(self):
        pass

    @abstractmethod
    def generate_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_instance(self, inst_path, data):
        pass

    def generate_raw_data_worker(self, rank):
        env = get_env(n_objs=self.cfg.prob.n_vars)
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"{rank}/1/10: Fetching instance data and order...")
            data = self.get_instance_data(self.cfg.size, self.cfg.split, pid)
            order = get_static_order(self.cfg.prob.name, self.cfg.prob.order_type, data)

            print(f"{rank}/2/10: Resetting env...")
            env.reset(self.cfg.prob.problem_type,
                      self.cfg.prob.preprocess,
                      self.cfg.prob.method,
                      self.cfg.prob.maximization,
                      self.cfg.prob.dominance,
                      self.cfg.prob.bdd_type,
                      self.cfg.prob.maxwidth,
                      order)

            print(f"{rank}/3/10: Initializing instance...")
            self.set_inst(env, data)

            print(f"{rank}/4/10: Preprocessing instance...")
            self.preprocess_inst(env)

            print(f"{rank}/5/10: Generating decision diagram...")
            env.initialize_dd_constructor()
            env.generate_dd()
            time_compile = env.get_time(CONST.TIME_COMPILE)

            print(f"{rank}/6/10: Fetching decision diagram...")
            start = time.time()
            dd = env.get_dd()
            time_fetch = time.time() - start

            exact_size = []
            for i, layer in enumerate(dd):
                exact_size.append(len(layer))
            dynamic_order = self.get_dynamic_order(env)

            print(f"{rank}/7/10: Computing Pareto Frontier...")
            try:
                signal.alarm(self.cfg.prob.time_limit)
                env.compute_pareto_frontier()
            except TimeoutError as exc:
                is_pf_computed = False
                print(f"PF not computed within {self.cfg.prob.time_limit} for pid {pid}")
            else:
                is_pf_computed = True
                print(f"PF computed successfully for pid {pid}")
            signal.alarm(0)
            if not is_pf_computed:
                continue
            time_pareto = env.get_time(CONST.TIME_PARETO)

            print(f"{rank}/8/10: Fetching Pareto Frontier...")
            frontier = env.get_frontier()

            print(f"{rank}/9/10: Marking Pareto nodes...")
            pareto_state_scores = self.get_pareto_state_scores(data, frontier["x"], order=dynamic_order)
            dd = self._tag_dd_nodes(dd, pareto_state_scores)

            print(f"{rank}/10/10: Saving data...")
            # Save order
            self.save_order(pid, dynamic_order)
            self.save_dd(pid, dd)
            self.save_solution(pid, frontier, dynamic_order)
            self.save_dm_stats(pid, frontier, env, time_fetch, time_compile, time_pareto)

    def generate_raw_data(self):
        if self.cfg.n_processes == 1:
            self.generate_raw_data_worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self.generate_raw_data_worker, args=(rank,)))

            for r in results:
                r.get()

    @abstractmethod
    def generate_dataset(self):
        pass

    @abstractmethod
    def get_instance_data(self, size, split, pid):
        pass

    @staticmethod
    def set_inst(env, data):
        env.set_inst(data['n_vars'],
                     data['n_cons'],
                     data['n_objs'],
                     data['obj_coeffs'],
                     data['cons_coeffs'],
                     data['rhs'])

    @staticmethod
    def preprocess_inst(env):
        pass

    def get_dynamic_order(self, env):
        pass

    @abstractmethod
    def get_pareto_state_scores(self, data, x, order=None):
        pass

    def save_order(self, pid, dynamic_order):
        if dynamic_order is not None:
            file_path = path.order / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
            file_path.mkdir(parents=True, exist_ok=True)
            file_path /= f"dynamic_{pid}.dat"
            with open(file_path, "w") as fp:
                fp.write(" ".join(map(str, dynamic_order)))

    def save_dd(self, pid, dd):
        file_path = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(dd, fp)

    def save_solution(self, pid, frontier, dynamic_order):
        file_path = path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            frontier["order"] = dynamic_order
            json.dump(frontier, fp)

    def save_dm_stats(self, pid, frontier, env, time_fetch, time_compile, time_pareto):
        file_path = path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.split, pid, len(frontier["z"]), env.initial_node_count,
                            env.initial_arcs_count, -1, time_fetch, time_compile, time_pareto]],
                          columns=["size", "split", "pid", "nnds", "inc", "iac", "Comp.", "fetch", "compilation",
                                   "pareto"])
        df.to_csv(file_path.parent / f"dm_stats_{pid}.csv", index=False)

    def _tag_dd_nodes(self, bdd, pareto_state_scores):
        assert len(pareto_state_scores) == len(bdd)
        items = np.array(range(0, 100))

        for l in range(len(bdd)):
            pareto_states, pareto_scores = pareto_state_scores[l]
            for pareto_state, score in zip(pareto_states, pareto_scores):
                _pareto_state = items[pareto_state.astype(bool)]
                is_found = False
                for n in bdd[l]:
                    if np.array_equal(n["s"], _pareto_state):
                        n["pareto"] = 1
                        n["score"] = score
                        is_found = True
                        break

                assert is_found

            for n in bdd[l]:
                if "pareto" not in n:
                    n["pareto"] = 0
                    n["score"] = 0

        return bdd
