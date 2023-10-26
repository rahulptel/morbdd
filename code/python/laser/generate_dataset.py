import multiprocessing as mp

import hydra
import numpy as np
import xgboost as xgb

from laser import resource_path
from laser.utils import convert_bdd_to_tensor_data
from laser.utils import get_instance_data
from laser.utils import get_order
from laser.utils import read_from_zip
from laser.utils import get_layer_weights
from laser.utils import convert_bdd_to_xgb_data
import shutil


def label_bdd(bdd, labeling_scheme):
    for l in range(len(bdd)):
        for n in bdd[l]:
            if labeling_scheme == "binary":
                n["l"] = 1 if n["pareto"] else 0
            elif labeling_scheme == "mo":
                # Margin one
                n["l"] = 1 if n["pareto"] else -1
            elif labeling_scheme == "mos":
                # Margin one score
                n["l"] = 1 + n["score"] if n["pareto"] else -1
            elif labeling_scheme == "nms":
                # Negative margin score
                n["l"] = n["score"] if n["pareto"] else -1
            else:
                raise ValueError("Invalid labeling scheme!")

    return bdd


def worker(rank, cfg):
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        # print(f"Processing pid {pid}...")
        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.label)

        print(f"\tRank {rank}: Converting BDD:{pid} to tensor dataset...")
        convert_bdd_to_tensor_data(cfg.prob,
                                   bdd=bdd,
                                   num_objs=cfg.num_objs,
                                   num_vars=cfg.num_vars,
                                   split=cfg.split,
                                   pid=pid,
                                   layer_penalty=cfg.layer_penalty,
                                   order_type=cfg.order_type,
                                   state_norm_const=cfg.state_norm_const,
                                   layer_norm_const=cfg.layer_norm_const,
                                   neg_pos_ratio=cfg.neg_pos_ratio,
                                   min_samples=cfg.min_samples,
                                   random_seed=cfg.seed)


def worker_xgb(rank, cfg):
    data_lst = []
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.label)

        data = convert_bdd_to_xgb_data(cfg.prob,
                                       bdd=bdd,
                                       num_objs=cfg.num_objs,
                                       num_vars=cfg.num_vars,
                                       split=cfg.split,
                                       pid=pid,
                                       layer_penalty=cfg.layer_penalty,
                                       order_type=cfg.order_type,
                                       state_norm_const=cfg.state_norm_const,
                                       layer_norm_const=cfg.layer_norm_const,
                                       neg_pos_ratio=cfg.neg_pos_ratio,
                                       min_samples=cfg.min_samples,
                                       random_seed=cfg.seed)
        data_lst.append(data)
        print(f"Processed {pid}...")

    data_lst = np.concatenate(data_lst)
    return data_lst


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    pool = mp.Pool(processes=cfg.num_processes)
    results = []
    if cfg.dtype == "Tensor":
        for rank in range(cfg.num_processes):
            results.append(pool.apply_async(worker, args=(rank, cfg)))

        for r in results:
            r.get()

    elif cfg.dtype == "DMatrix":
        for rank in range(cfg.num_processes):
            results.append(pool.apply_async(worker_xgb, args=(rank, cfg)))

        # a, b, c = worker_xgb(0, cfg)
        results = [r.get() for r in results]
        results = np.concatenate(results)

        file_path = resource_path / f"xgb_dataset/{cfg.prob}/{cfg.size}"
        file_path.mkdir(parents=True, exist_ok=True)
        name = f"{cfg.split}-{cfg.from_pid}-{cfg.to_pid}"
        if cfg.flag_layer_penalty:
            name += f"-{cfg.layer_penalty}"
        if cfg.flag_label_penalty:
            name += f"-{cfg.label_penalty}"
        name += ".npy"
        with open(name, "wb") as fp:
            np.save(fp, results)


if __name__ == '__main__':
    main()
