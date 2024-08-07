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
from morbdd.utils import get_dataset_path


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
    def _get_instance_data(self, *args):
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
        env.preprocess_inst()

    @staticmethod
    def _reduce_dd(env):
        pass

    def _get_static_order(self, *args):
        return []

    @staticmethod
    def _get_dynamic_order(env):
        return []

    @abstractmethod
    def _get_pareto_state_scores(self, data, x, order=None):
        pass

    def _save_order(self, pid, order):
        file_path = path.order / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{pid}.dat"
        with open(file_path, "w") as fp:
            fp.write(" ".join(map(str, order)))

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
        env = get_env(n_objs=self.cfg.prob.n_objs)
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            order_type = None
            print(f"{rank}/1/10: Fetching instance data and order...")
            data = self._get_instance_data(pid)
            static_order = self._get_static_order(data)
            if len(static_order):
                self._save_order(pid, static_order)
                order_type = "static"

            print(f"{rank}/2/10: Resetting env...")
            env.reset(self.cfg.prob.problem_type,
                      self.cfg.prob.preprocess,
                      self.cfg.prob.pf_enum_method,
                      self.cfg.prob.maximization,
                      self.cfg.prob.dominance,
                      self.cfg.prob.bdd_type,
                      self.cfg.prob.maxwidth,
                      static_order)

            print(f"{rank}/3/10: Initializing instance...")
            self._set_inst(env, data)

            print(f"{rank}/4/10: Preprocessing instance...")
            self._preprocess_inst(env)

            print(f"{rank}/5/10: Generating decision diagram...")
            env.initialize_dd_constructor()
            env.generate_dd()
            self._reduce_dd(env)
            time_compile = env.get_time(CONST.TIME_COMPILE)

            print(f"{rank}/6/10: Fetching decision diagram...")
            start = time.time()
            dd = env.get_dd()
            time_fetch = time.time() - start

            exact_size = []
            for i, layer in enumerate(dd):
                exact_size.append(len(layer))
            dynamic_order = self._get_dynamic_order(env)
            if len(dynamic_order):
                self._save_order(pid, dynamic_order)
                order_type = "dynamic"

            print(f"{rank}/7/10: Computing Pareto Frontier...")
            try:
                signal.alarm(self.cfg.prob.time_limit)
                env.compute_pareto_frontier()
            except TimeoutError:
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
            print(f"{pid}: |Z| = {len(frontier['z'])}")
            print(f"{rank}/9/10: Marking Pareto nodes...")
            pareto_state_scores = self._get_pareto_state_scores(data, frontier["x"], order=dynamic_order)
            dd = self._tag_dd_nodes(dd, pareto_state_scores)

            print(f"{rank}/10/10: Saving data...")
            # Save dd, solution and stats
            self._save_dd(pid, dd)
            sol_var_order = static_order if order_type == "static" else dynamic_order
            self._save_solution(pid, frontier, sol_var_order)
            self._save_dm_stats(pid, frontier, env, time_fetch, time_compile, time_pareto)

    @abstractmethod
    def _get_bdd_node_dataset(self, *args):
        pass

    def _generate_dataset_worker(self, rank, dataset_path):
        archive_bdds = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}.zip"

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            # Read instance data
            inst_data = self._get_instance_data(pid)
            file = f"{self.cfg.prob.size}/{self.cfg.split}/{pid}.json"
            bdd = read_from_zip(archive_bdds, file, format="json")
            # Read order
            order = path.order.joinpath(
                f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}/{pid}.dat").read_text()
            order = np.array(list(map(int, order.strip().split())))
            # Get node data
            self._get_bdd_node_dataset(pid, inst_data, order, bdd, dataset_path)

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
                    inst_path.parent.mkdir(parents=True, exist_ok=True)
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
        dataset_path = get_dataset_path(self.cfg)
        dataset_path.mkdir(exist_ok=True, parents=True)

        if self.cfg.n_processes == 1:
            self._generate_dataset_worker(0, dataset_path)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self._generate_dataset_worker, args=(rank, dataset_path)))

            for r in results:
                r.get()

        M = None
        for p in dataset_path.rglob("*.npy"):
            mat = np.load(p)
            M = np.concatenate((M, mat), axis=0) if M is not None else mat

        prefix = dataset_path.stem
        np.save(dataset_path.parent / f"{prefix}-{self.cfg.split}.npy", M)
