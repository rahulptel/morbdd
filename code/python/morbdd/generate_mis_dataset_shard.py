import multiprocessing as mp
import random

import hydra
import numpy as np
import torch
import tarfile
from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip
from morbdd.utils import get_layer_weights
from pathlib import Path
import json
import os


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def get_node_data(pid, bdd, order):
    dataset = []
    for lid, layer in enumerate(bdd):
        data = {"pid": pid, "lid": lid, "vid": int(order[lid + 1]), "pos": [], "neg": []}
        for nid, node in enumerate(layer):
            # Binary state of the current node
            # state = np.zeros(cfg.prob.n_vars)
            # state[node['s']] = 1
            # node_data = np.concatenate(([lid],
            #                             state))
            if node['score'] > 0:
                data["pos"].append(node["s"])
            else:
                data["neg"].append(node["s"])
        dataset.append(data)

    return dataset


def worker(rank, cfg):
    archive_bdds = path.bdd / f"{cfg.prob.name}/{cfg.size}.zip"
    samples = []
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.n_processes):
        # Read instance data
        # data = get_instance_data(cfg.prob.name, cfg.size, cfg.split, pid)
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive_bdds, file, format="json")
        # Read order
        order = path.order.joinpath(f"{cfg.prob.name}/{cfg.size}/{cfg.split}/{pid}.dat").read_text()
        order = np.array(list(map(int, order.strip().split())))
        # Get node data
        samples = get_node_data(pid, bdd, order)
        for lid, sample in enumerate(samples):
            json.dump(sample, open(f"../../resources/datasets/indepset/{pid}-{lid}.json", "w"))


@hydra.main(config_path="configs", config_name="mis_dataset", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    if cfg.n_processes == 1:
        worker(0, cfg)
    else:
        pool = mp.Pool(processes=cfg.n_processes)
        results = []

        for rank in range(cfg.n_processes):
            results.append(pool.apply_async(worker, args=(rank, cfg)))

        for r in results:
            r.get()

    root = Path("../../resources/datasets/indepset/")
    count = 0
    filename = ""
    for pid in range(cfg.from_pid, cfg.to_pid):
        if count == 0:
            filename = "bdd-layer-" + str(pid)

        files = list(root.rglob(f"{pid}-*.json"))
        with tarfile.open(root.joinpath(f"{filename}.tar"), 'a') as tar:
            for file_name in files:
                tar.add(file_name, arcname=file_name.name)

        for file_name in files:
            os.remove(file_name)

        count += 1
        if count == 10:
            count = 0


if __name__ == "__main__":
    main()
