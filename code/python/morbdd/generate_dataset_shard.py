import json
import multiprocessing as mp
import os
import random
import tarfile

import hydra
import numpy as np

from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}_{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def get_node_data_indepset(pid, bdd, order):
    dataset = []
    for lid, layer in enumerate(bdd):
        data = {"pid": pid, "lid": lid, "vid": int(order[lid + 1]), "pos": [], "neg": []}
        for nid, node in enumerate(layer):
            if node['score'] > 0:
                data["pos"].append(list(map(int, node["s"])))
            else:
                data["neg"].append(list(map(int, node["s"])))
        dataset.append(data)

    return dataset


def get_node_data_knapsack(pid, bdd, order, capacity):
    dataset = []
    for lid, layer in enumerate(bdd):
        data = {"pid": pid, "lid": lid, "vid": int(order[lid + 1]), "pos": [], "neg": []}
        for nid, node in enumerate(layer):
            if node['score'] > 0:
                data["pos"].append(node["s"][0] / capacity)
            else:
                data["neg"].append(node["s"][0] / capacity)
        dataset.append(data)

    return dataset


def get_node_data(problem, pid, bdd, order, capacity=None):
    if "indepset" in problem:
        return get_node_data_indepset(pid, bdd, order)
    elif "knapsack" in problem:
        return get_node_data_knapsack(pid, bdd, order, capacity)


def worker(rank, prob_name, size, split, from_pid, to_pid, n_processes):
    archive_bdds = path.bdd / f"{prob_name}/{size}.zip"
    archive_order = path.order / f"{prob_name}/{size}.zip"
    root = path.dataset / f"{prob_name}/{size}/{split}"
    root.mkdir(exist_ok=True, parents=True)
    for pid in range(from_pid + rank, to_pid, n_processes):
        # Read instance data
        capacity = None
        if prob_name == "knapsack":
            data = get_instance_data(prob_name, size, split, pid)
            capacity = data["capacity"]
        # Read BDD
        file = f"{size}/{split}/{pid}.json"
        bdd = read_from_zip(archive_bdds, file, format="json")
        # Read order
        file = f"{size}/{split}/{pid}.dat"
        order = read_from_zip(archive_order, file)
        order = order.read().decode("utf-8")
        # order = path.order.joinpath(f"{prob_name}/{size}/{split}/{pid}.dat").read_text()
        order = np.array(list(map(int, order.strip().split())))
        # Get node data
        samples = get_node_data(prob_name, pid, bdd, order, capacity=capacity)
        for lid, sample in enumerate(samples):
            json.dump(sample, open(f"{root}/{pid}-{lid}.json", "w"))


@hydra.main(config_path="configs", config_name="dataset_shard", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    pool = mp.Pool(processes=cfg.n_processes) if cfg.n_processes > 1 else None
    root = path.dataset / f"{cfg.prob.name}/{cfg.size}/{cfg.split}"
    shard_counter = 0
    for pid in range(cfg.from_pid, cfg.to_pid, cfg.shard_size):
        # Generate jsons
        if cfg.n_processes == 1:
            worker(0, cfg.prob.name, cfg.size, cfg.split, pid, pid + cfg.shard_size, 1)
        else:
            results = []
            for rank in range(cfg.n_processes):
                results.append(pool.apply_async(worker, args=(rank, cfg.prob.name, cfg.size, cfg.split,
                                                              pid, pid + 10, cfg.n_processes)))

            for r in results:
                r.get()

        tarname = f"bdd-layer-{shard_counter}"
        files = []
        for i in range(pid, pid + cfg.shard_size):
            files.extend(list(root.rglob(f"{i}-*.json")))

        new_names = []
        for file in files:
            new_name = f"{random.randint(0, 10000)}-" + file.name
            new_name = file.parent / new_name
            file.rename(new_name)
            new_names.append(new_name)

        with tarfile.open(root.joinpath(f"{tarname}.tar"), 'a') as tar:
            for file in new_names:
                tar.add(file, arcname=file.name)
        for file in new_names:
            os.remove(file)
        shard_counter += 1


if __name__ == "__main__":
    main()
