import json
import multiprocessing as mp
import signal
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from morbdd import CONST
from morbdd import ResourcePaths as path
from morbdd.utils import get_env
from morbdd.utils import get_static_order
from morbdd.utils import handle_timeout
from morbdd.utils import read_from_zip
from morbdd.utils import zipdir
import zipfile
import shutil


class DataManager(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def _get_instance_path(self, *args):
        pass

    @abstractmethod
    def _generate_instance(self, *args, **kwargs):
        pass

    @abstractmethod
    def _save_instance(self, inst_path, data):
        pass

    @abstractmethod
    def _get_instance_data(self, size, split, pid):
        pass

    @staticmethod
    def _set_inst(env, data):
        env.set_inst(data['n_vars'],
                     data['n_cons'],
                     data['n_objs'],
                     data['obj_coeffs'],
                     data['cons_coeffs'],
                     data['rhs'])

    @staticmethod
    def _preprocess_inst(env):
        pass

    @abstractmethod
    def _get_pareto_state_scores(self, data, x, order=None):
        pass

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

    def _save_order(self, pid, dynamic_order):
        if dynamic_order is not None:
            file_path = path.order / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
            file_path.mkdir(parents=True, exist_ok=True)
            file_path /= f"dynamic_{pid}.dat"
            with open(file_path, "w") as fp:
                fp.write(" ".join(map(str, dynamic_order)))

    def _save_dd(self, pid, dd):
        file_path = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            json.dump(dd, fp)

    def _save_solution(self, pid, frontier, dynamic_order):
        file_path = path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.json"
        with open(file_path, "w") as fp:
            frontier["order"] = dynamic_order
            json.dump(frontier, fp)

    def _save_dm_stats(self, pid, frontier, env, time_fetch, time_compile, time_pareto):
        file_path = path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        df = pd.DataFrame([[self.cfg.prob.size, self.cfg.split, pid, len(frontier["z"]), env.initial_node_count,
                            env.initial_arcs_count, -1, time_fetch, time_compile, time_pareto]],
                          columns=["size", "split", "pid", "nnds", "inc", "iac", "Comp.", "fetch", "compilation",
                                   "pareto"])
        df.to_csv(file_path.parent / f"dm_stats_{pid}.csv", index=False)

    def _generate_bdd_data_worker(self, rank):
        env = get_env(n_objs=self.cfg.prob.n_vars)
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"{rank}/1/10: Fetching instance data and order...")
            data = self._get_instance_data(self.cfg.size, self.cfg.split, pid)
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
            self._set_inst(env, data)

            print(f"{rank}/4/10: Preprocessing instance...")
            self._preprocess_inst(env)

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
            dynamic_order = env.get_var_layer()

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
            pareto_state_scores = self._get_pareto_state_scores(data, frontier["x"], order=dynamic_order)
            dd = self._tag_dd_nodes(dd, pareto_state_scores)

            print(f"{rank}/10/10: Saving data...")
            # Save order
            self._save_order(pid, dynamic_order)
            self._save_dd(pid, dd)
            self._save_solution(pid, frontier, dynamic_order)
            self._save_dm_stats(pid, frontier, env, time_fetch, time_compile, time_pareto)

    @abstractmethod
    def _get_bdd_node_dataset(self, *args):
        pass

    def _generate_dataset_worker(self, rank):
        archive_bdds = path.bdd / f"{self.cfg.prob.name}/{self.cfg.size}.zip"
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            # Read instance data
            inst_data = self._get_instance_data(self.cfg.size, self.cfg.split, pid)
            file = f"{self.cfg.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive_bdds, file, format="json")
            # Read order
            order = path.order.joinpath(
                f"{self.cfg.prob.name}/{self.cfg.size}/{self.cfg.split}/{pid}.dat").read_text()
            order = np.array(list(map(int, order.strip().split())))
            # Get node data
            self._get_bdd_node_dataset(pid, inst_data, order, bdd)

    def _generate_dataset_tensor(self, rank):
        pass

    def _generate_dataset_dmatrix(self, rank):
        pass

    def generate_instances(self):
        rng = np.random.RandomState(self.cfg.seed)

        for s in self.cfg.sizes:
            n_objs, n_vars = map(int, s.split("-"))
            start, end = 0, self.cfg.n_train
            for split in ["train", "val", "test"]:
                for pid in range(start, end):
                    inst_path = self._get_instance_path(self.cfg.seed, n_objs, n_vars, split, pid)
                    inst_data = self._generate_instance(rng, n_vars, n_objs)
                    self._save_instance(inst_path, inst_data)

                if split == "train":
                    start = self.cfg.n_train
                    end = start + self.cfg.n_val
                elif split == "val":
                    start = self.cfg.n_train + self.cfg.n_val
                    end = start + self.cfg.n_test

            with zipfile.ZipFile(str(inst_path.parent.parent) + ".zip", "w", zipfile.ZIP_DEFLATED) as zf:
                zipdir(inst_path.parent.parent, zf)
            shutil.rmtree(inst_path.parent.parent)

    def generate_bdd_data(self):
        if self.cfg.n_processes == 1:
            self._generate_bdd_data_worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self._generate_bdd_data_worker, args=(rank,)))

            for r in results:
                r.get()

    def generate_dataset(self):
        if self.cfg.n_processes == 1:
            self._generate_dataset_worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self._generate_dataset_worker, args=(rank,)))

            for r in results:
                r.get()

        # dataset_path = path.dataset / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        # M = None
        # for p in dataset_path.rglob("*.npy"):
        #     mat = np.load(p)
        #     M = np.concatenate((M, mat), axis=0) if M is not None else mat
        #
        # np.save(dataset_path.parent / f"{self.cfg.split}.npy", M)
