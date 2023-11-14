from pathlib import Path
import json
import pandas as pd
from laser import resource_path
import zipfile
import numpy as np
from laser.utils import get_instance_data
from laser.utils import get_order
import copy
import multiprocessing as mp
import hydra

from laser.utils import get_xgb_model_name
import hashlib


def call_get_model_name(cfg):
    return get_xgb_model_name(max_depth=cfg.max_depth,
                              eta=cfg.eta,
                              objective=cfg.objective,
                              num_round=cfg.num_round,
                              early_stopping_rounds=cfg.early_stopping_rounds,
                              evals=cfg.evals,
                              eval_metric=cfg.eval_metric,
                              seed=cfg.seed,
                              prob_name=cfg.prob.name,
                              num_objs=cfg.prob.num_objs,
                              num_vars=cfg.prob.num_vars,
                              order=cfg.prob.order,
                              layer_norm_const=cfg.prob.layer_norm_const,
                              state_norm_const=cfg.prob.state_norm_const,
                              train_from_pid=cfg.train.from_pid,
                              train_to_pid=cfg.train.to_pid,
                              train_neg_pos_ratio=cfg.train.neg_pos_ratio,
                              train_min_samples=cfg.train.min_samples,
                              train_flag_layer_penalty=cfg.train.flag_layer_penalty,
                              train_layer_penalty=cfg.train.layer_penalty,
                              train_flag_imbalance_penalty=cfg.train.flag_imbalance_penalty,
                              train_flag_importance_penalty=cfg.train.flag_importance_penalty,
                              train_penalty_aggregation=cfg.train.penalty_aggregation,
                              val_from_pid=cfg.val.from_pid,
                              val_to_pid=cfg.val.to_pid,
                              val_neg_pos_ratio=cfg.val.neg_pos_ratio,
                              val_min_samples=cfg.val.min_samples,
                              val_flag_layer_penalty=cfg.val.flag_layer_penalty,
                              val_layer_penalty=cfg.val.layer_penalty,
                              val_flag_imbalance_penalty=cfg.val.flag_imbalance_penalty,
                              val_flag_importance_penalty=cfg.val.flag_importance_penalty,
                              val_penalty_aggregation=cfg.val.penalty_aggregation,
                              device=cfg.device)


def count_ndps(true_pf, pred_pf, i, mdl_hex):
    z, z_pred = np.array(true_pf), np.array(pred_pf)
    print(i, z.shape, z_pred.shape)
    assert z.shape[1] == z_pred.shape[1]
    num_objs = z.shape[1]

    z = z[np.lexsort([z[:, num_objs - i - 1] for i in range(num_objs)])]
    z_pred = z_pred[np.lexsort([z_pred[:, num_objs - i - 1] for i in range(num_objs)])]

    j_prev = 0
    counter = 0
    found_ndps = []
    for i in range(z.shape[0]):
        # print(i, counter)
        j = copy.deepcopy(j_prev)
        while j < z_pred.shape[0]:
            if np.array_equal(z[i], z_pred[j]):
                counter += 1
                j_prev = copy.deepcopy(j + 1)
                found_ndps.append(z[i])
                break

            j += 1

    found_ndps = np.array(found_ndps)
    p = resource_path / f"predictions/xgb/{mdl_hex}/count_pareto"
    p.mkdir(exist_ok=True, parents=True)
    with open(resource_path / f"predictions/xgb/{mdl_hex}/count_pareto/found_ndps_{i}.npy", "wb") as fp:
        np.save(fp, found_ndps)
    #
    # with open("true_ndps.npy", "wb") as fp:
    #     np.save(fp, z)
    #
    # with open("pred_ndps.npy", "wb") as fp:
    #     np.save(fp, z_pred)
    return counter


def worker(i, cfg, mdl_hex):
    # Load predicted solution
    sols_pred_path = resource_path / f"predictions/xgb/{mdl_hex}/sols_pred"
    sol_path = sols_pred_path / f"sol_{i}.json"
    if sol_path.exists():
        with open(sols_pred_path / f"sol_{i}.json", "r") as fp:
            sol_pred = json.load(fp)

        # inst_data = get_instance_data("knapsack", "7_40", cfg.deploy.split, i)
        # order = get_order("knapsack", "MinWt", inst_data)
        # weight = np.array(inst_data["weight"])[order]
        # num_nodes = get_node_count(sol_pred["x"], weight)
        zfp = zipfile.Path(resource_path / f"sols/knapsack/7_40.zip")
        if zfp.joinpath(f"7_40/{cfg.deploy.split}/{i}.json"):
            zf = zipfile.ZipFile(resource_path / f"sols/knapsack/7_40.zip")
            with zf.open(f"7_40/{cfg.deploy.split}/{i}.json") as fp:
                sol = json.load(fp)

            ndps_in_pred = count_ndps(sol["z"], sol_pred["z"], i, mdl_hex)
            num_pred_ndps = len(sol_pred["z"])
            num_total_ndps = len(sol["z"])
            frac_true_ndps_in_pred = ndps_in_pred / num_pred_ndps
            frac_ndps_recovered = ndps_in_pred / num_total_ndps
            print(f"Result for inst {i}: ",
                  ndps_in_pred,
                  frac_true_ndps_in_pred,
                  frac_ndps_recovered)
            count_pareto_file = resource_path / f"predictions/xgb/{mdl_hex}/count_pareto/{i}.txt"
            count_pareto_file.parent.mkdir(exist_ok=True, parents=True)
            count_pareto_file.write_text(
                f"{ndps_in_pred}, {num_pred_ndps}, {num_total_ndps}, {frac_true_ndps_in_pred}, {frac_ndps_recovered}")


@hydra.main(version_base="1.2", config_path="./configs", config_name="post_process.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()
    print("Mdl hex", mdl_hex)

    pool = mp.Pool(processes=cfg.nthread)
    results = [pool.apply_async(worker, args=(i, cfg, mdl_hex)) for i in range(cfg.deploy.from_pid, cfg.deploy.to_pid)]
    results = [r.get() for r in results]

    # worker(1000, cfg, mdl_hex)


if __name__ == '__main__':
    main()
