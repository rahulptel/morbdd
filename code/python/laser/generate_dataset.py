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
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.num_processes):
        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive, file, format="json")
        if bdd is None:
            continue
        bdd = label_bdd(bdd, cfg.label)

        convert_bdd_to_xgb_data(cfg.prob,
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
        print(f"Processed {pid}...")


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    worker_fn = None
    if cfg.dtype == "Tensor":
        worker_fn = worker
    elif cfg.dtype == "DMatrix":
        worker_fn = worker_xgb
    else:
        raise ValueError("Invalid dataset type!")

    pool = mp.Pool(processes=cfg.num_processes)
    results = []
    for rank in range(cfg.num_processes):
        results.append(pool.apply_async(worker_fn, args=(rank, cfg)))

    for r in results:
        r.get()


if __name__ == '__main__':
    main()
