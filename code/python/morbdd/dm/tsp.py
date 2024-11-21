import json
import multiprocessing as mp
import random
import signal
import time

import numpy as np
import pandas as pd

from morbdd import ResourcePaths as path
from morbdd.utils import handle_timeout
from morbdd.utils.tsp import get_env
from morbdd.utils.tsp import get_instance_data
from .dm import DataManager

random.seed(7)


class TSPDataManager(DataManager):
    def _save_states_dataset(self, pid, pareto_states):
        parent = (
            path.dataset / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        np.savez(f"{parent}/ps_{pid}.npz", pareto_states)

    def _save_solution(self, pid, frontier):
        parent = (
            path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        np.savez(f"{parent}/sol_{pid}.npz", x=frontier["x"], z=frontier["z"])

    def _save_dm_stats(self, pid, frontier, env, time_fetch, time_compile, time_pareto):
        file_path = (
            path.sol / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        )
        file_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                [
                    self.cfg.prob.size,
                    self.cfg.split,
                    pid,
                    len(frontier["z"]),
                    time_compile,
                    time_pareto,
                ]
            ],
            columns=["size", "split", "pid", "nnds", "compilation", "pareto"],
        )
        df.to_csv(file_path.parent / f"dm_stats_{pid}.csv", index=False)

    def _get_instance_path(self, seed, n_objs, n_vars, split, pid):
        size = f"{n_objs}_{n_vars}"
        return path.inst / f"tsp/{size}/{split}/tsp_{seed}_{size}_{pid}.npz"

    def _generate_instance(self, rng, n_vars, n_objs):
        # List to store distance matrices
        distance_matrices = []
        coordinate_matrices = []

        # Generate p sets of coordinates
        for _ in range(n_objs):
            # Random integer coordinates for each city
            coordinates = np.random.randint(
                0, self.cfg.prob.grid_size, size=(n_vars, 2)
            )
            coordinate_matrices.append(coordinates)

            # Calculate the distance matrix (Euclidean distances between cities)
            distances = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    dist = int(np.linalg.norm(coordinates[i] - coordinates[j]))
                    distances[i, j] = dist
                    distances[j, i] = dist  # symmetric matrix

            # Append the distance matrix to the list
            distance_matrices.append(distances)

        return {"coords": coordinate_matrices, "dists": distance_matrices}

    def _save_instance(self, inst_path, data):
        np.savez(str(inst_path), coords=data["coords"], dists=data["dists"])

    def _get_instance_data(self, pid):
        size = f"{self.cfg.prob.n_objs}_{self.cfg.prob.n_vars}"
        data = get_instance_data(size, self.cfg.split, pid)

        return data

    def _get_pareto_state_scores(self, x, order=None):
        x = np.array(x)
        x = x[:, 1:]
        n_pareto_sol = x.shape[0]

        pareto_state_scores = []
        for i in range(1, x.shape[1] + 1):
            x_ = x[:, :i]
            states = np.zeros((x.shape[0], self.cfg.prob.n_vars))
            ind = np.arange(states.shape[0])
            for j in range(x_.shape[1]):
                states[ind, x_[:, j]] = 1
            last_city = x_[:, j]

            states = np.hstack((states, last_city.reshape(-1, 1)))
            states_uq, cnt = np.unique(states, axis=0, return_counts=True)
            cntn = cnt / n_pareto_sol
            pareto_state_scores.append((states_uq, cntn))

        return pareto_state_scores

    def _tag_dd_nodes(self, pid, dd, pareto_state_scores):
        assert len(pareto_state_scores) == len(dd)

        tick = time.time()
        result = []
        for l in range(len(dd)):
            pareto_states, pareto_scores = pareto_state_scores[l]
            pos, neg = [], []
            for pareto_state, score in zip(pareto_states, pareto_scores):
                for nid, n in enumerate(dd[l]):
                    if np.array_equal(pareto_state, n):
                        pos.append([pid, l, nid, float(score), 1])
                        break

            all_ids = [i for i in range(len(dd[l]))]
            pos_ids = [p[2] for p in pos]
            neg_ids = set(all_ids).difference(set(pos_ids))
            for nid in neg_ids:
                neg.append([pid, l, nid, 0, 0])

            result.extend(pos)
            random.shuffle(neg)
            neg_upto = min(len(neg), 4 * len(pos))
            if neg_upto > 0:
                result.extend(neg[:neg_upto])

        return result

    def _generate_dd_data_worker(self, rank):
        env = get_env(n_objs=self.cfg.prob.n_objs)
        signal.signal(signal.SIGALRM, handle_timeout)

        for pid in range(
            self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes
        ):
            print(f"{rank}/1/10: Fetching instance data and order...")
            data = self._get_instance_data(pid)

            env.reset()
            env.set_inst(
                self.cfg.prob.n_vars,
                self.cfg.prob.n_objs,
                data["dists"].astype(int).tolist(),
            )
            env.initialize_dd_constructor()
            tick = time.time()
            env.generate_dd()
            time_compile = time.time() - tick
            exact_dd = env.get_dd()

            print(f"{rank}/2/10: Computing Pareto Frontier...")
            time_pareto = 1800
            try:
                signal.alarm(self.cfg.prob.time_limit)
                tick = time.time()
                env.compute_pareto_frontier()
                time_pareto = time.time() - tick
            except TimeoutError:
                is_pf_computed = False
                print(
                    f"PF not computed within {self.cfg.prob.time_limit} for pid {pid}"
                )
            else:
                is_pf_computed = True
                print(f"PF computed successfully for pid {pid}")
            signal.alarm(0)
            if not is_pf_computed:
                continue

            print(f"{rank}/8/10: Fetching Pareto Frontier...")
            frontier = env.get_frontier()
            print(f"{pid}: |Z| = {len(frontier['z'])}")

            print(f"{rank}/10/10: Saving data...")
            # Save dd, solution and stats
            self._save_dd(pid, exact_dd)
            self._save_solution(pid, frontier)
            self._save_dm_stats(pid, frontier, env, -1, time_compile, time_pareto)

    def generate_dd_dataset(self):
        if self.cfg.n_processes == 1:
            self._generate_dd_data_worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(
                    pool.apply_async(self._generate_dd_data_worker, args=(rank,))
                )

            for r in results:
                r.get()

    def _get_bdd_node_dataset(self, pid, inst_data, dd, dataset_path):
        sol_path = (
            path.sol
            / self.cfg.prob.name
            / self.cfg.prob.sizes
            / self.cfg.splits
            / f"sol_{pid}.npz"
        )
        if sol_path.exists():
            frontier = np.load(sol_path)
            pareto_state_scores = self._get_pareto_state_scores(frontier["x"])
            dataset = self._tag_dd_nodes(pid, dd, pareto_state_scores)
            dataset = np.array(dataset)
            print(pid, dataset.shape)
            self._save_states_dataset(pid, dataset)

    def _generate_dataset_worker(self, rank, dataset_path):
        dd_path = path.bdd / f"{self.cfg.prob.name}/{self.cfg.prob.size}/tsp_dd.json"

        for pid in range(
            self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes
        ):
            print("Processing pid {}".format(pid))
            # Read instance data
            inst_data = self._get_instance_data(pid)
            dd = json.load(open(dd_path, "r"))
            # Get node data
            self._get_bdd_node_dataset(pid, inst_data, dd, dataset_path)

    def generate_dataset(self):
        dataset_path = (
            path.dataset / f"{self.cfg.prob.name}/{self.cfg.prob.size}/{self.cfg.split}"
        )
        dataset_path.mkdir(exist_ok=True, parents=True)

        if self.cfg.n_processes == 1:
            self._generate_dataset_worker(0, dataset_path)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(
                    pool.apply_async(
                        self._generate_dataset_worker, args=(rank, dataset_path)
                    )
                )

            for r in results:
                r.get()

        if self.cfg.concat:
            print("Concatenating files...")
            M = None
            for p in dataset_path.rglob("*.npz"):
                mat = np.load(p)
                M = np.concatenate((M, mat), axis=0) if M is not None else mat
            print(f"Dataset size (all instances concat): {M.shape}")

            prefix = dataset_path.stem
            np.save(dataset_path.parent / f"{prefix}-{self.cfg.split}.npz", M)
