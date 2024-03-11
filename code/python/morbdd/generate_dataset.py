import multiprocessing as mp

import hydra

from laser import resource_path
from laser.utils import convert_bdd_to_tensor_data
from laser.utils import convert_bdd_to_xgb_data
from laser.utils import convert_bdd_to_xgb_mixed_data
from laser.utils import label_bdd
from laser.utils import read_from_zip


def worker_nn(rank, cfg):
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
                                   order_type=cfg.order_type,
                                   state_norm_const=cfg.state_norm_const,
                                   layer_norm_const=cfg.layer_norm_const,
                                   task=cfg.task,
                                   label_type=cfg.label,
                                   neg_pos_ratio=cfg.neg_pos_ratio,
                                   min_samples=cfg.min_samples,
                                   flag_layer_penalty=cfg.flag_layer_penalty,
                                   layer_penalty=cfg.layer_penalty,
                                   flag_imbalance_penalty=cfg.flag_imbalance_penalty,
                                   flag_importance_penalty=cfg.flag_importance_penalty,
                                   penalty_aggregation=cfg.penalty_aggregation,
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
                                order_type=cfg.order_type,
                                state_norm_const=cfg.state_norm_const,
                                layer_norm_const=cfg.layer_norm_const,
                                task=cfg.task,
                                label_type=cfg.label,
                                neg_pos_ratio=cfg.neg_pos_ratio,
                                min_samples=cfg.min_samples,
                                flag_layer_penalty=cfg.flag_layer_penalty,
                                layer_penalty=cfg.layer_penalty,
                                flag_imbalance_penalty=cfg.flag_imbalance_penalty,
                                flag_importance_penalty=cfg.flag_importance_penalty,
                                penalty_aggregation=cfg.penalty_aggregation,
                                random_seed=cfg.seed)


def worker_xgb_mixed(rank, cfg):
    sizes = str(cfg.mixed.sizes)
    sizes = sizes.strip().split(",")
    counter = int(cfg.from_pid)
    for size in sizes:
        num_objs, num_vars = list(map(int, size.split("-")))
        cfg.size = f"{num_objs}_{num_vars}"
        archive = resource_path / f"bdds/{cfg.prob}/{cfg.size}.zip"
        for pid in range(cfg.from_pid, cfg.to_pid):
            file = f"{cfg.size}/{cfg.split}/{pid}.json"
            bdd = read_from_zip(archive, file, format="json")
            print(archive, pid)
            if bdd is None:
                continue
            bdd = label_bdd(bdd, cfg.label)

            convert_bdd_to_xgb_mixed_data(cfg.prob,
                                          counter=counter,
                                          bdd=bdd,
                                          num_objs=num_objs,
                                          num_vars=num_vars,
                                          split=cfg.split,
                                          pid=pid,
                                          order_type=cfg.order_type,
                                          state_norm_const=cfg.state_norm_const,
                                          layer_norm_const=cfg.layer_norm_const,
                                          task=cfg.task,
                                          label_type=cfg.label,
                                          neg_pos_ratio=cfg.neg_pos_ratio,
                                          min_samples=cfg.min_samples,
                                          flag_layer_penalty=cfg.flag_layer_penalty,
                                          layer_penalty=cfg.layer_penalty,
                                          flag_imbalance_penalty=cfg.flag_imbalance_penalty,
                                          flag_importance_penalty=cfg.flag_importance_penalty,
                                          penalty_aggregation=cfg.penalty_aggregation,
                                          random_seed=cfg.seed)
            counter += 1


@hydra.main(version_base="1.2", config_path="./configs", config_name="bdd_dataset.yaml")
def main(cfg):
    worker_fn = None
    if cfg.dtype == "Tensor":
        worker_fn = worker_nn
    elif cfg.dtype == "DMatrix":
        worker_fn = worker_xgb
    elif cfg.dtype == "DMatrix-mixed":
        worker_fn = worker_xgb_mixed
    else:
        raise ValueError("Invalid dataset type!")

    # pool = mp.Pool(processes=cfg.num_processes)
    # results = []
    # for rank in range(cfg.num_processes):
    #     results.append(pool.apply_async(worker_fn, args=(rank, cfg)))
    #
    # for r in results:
    #     r.get()
    # Debug
    worker_fn(0, cfg)


if __name__ == '__main__':
    main()
