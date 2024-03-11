import copy
import hashlib
import json
import multiprocessing as mp
import zipfile

import hydra
import numpy as np

from morbdd import resource_path
from morbdd.utils import get_xgb_model_name


def call_get_model_name(cfg):
    return get_xgb_model_name(max_depth=cfg.max_depth,
                              eta=cfg.eta,
                              min_child_weight=cfg.min_child_weight,
                              subsample=cfg.subsample,
                              colsample_bytree=cfg.colsample_bytree,
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


# def find_ndps_in_preds(true_pf, pred_pf, i, mdl_hex):
#     z, z_pred = np.array(true_pf), np.array(pred_pf)
#     assert z.shape[1] == z_pred.shape[1]
#     num_objs = z.shape[1]
#
#     z = z[np.lexsort([z[:, num_objs - i - 1] for i in range(num_objs)])]
#     z_pred = z_pred[np.lexsort([z_pred[:, num_objs - i - 1] for i in range(num_objs)])]
#
#     j_prev = 0
#     counter = 0
#     found_ndps = []
#     for i in range(z.shape[0]):
#         # print(i, counter)
#         j = copy.deepcopy(j_prev)
#         while j < z_pred.shape[0]:
#             if np.array_equal(z[i], z_pred[j]):
#                 counter += 1
#                 j_prev = copy.deepcopy(j + 1)
#                 found_ndps.append(z[i])
#                 break
#
#             j += 1
#
#     found_ndps = np.array(found_ndps)
#     # p = resource_path / f"predictions/xgb/{mdl_hex}/count_pareto"
#     # p.mkdir(exist_ok=True, parents=True)
#     # with open(resource_path / f"predictions/xgb/{mdl_hex}/count_pareto/found_ndps_{i}.npy", "wb") as fp:
#     #     np.save(fp, found_ndps)
#     #
#     # with open("true_ndps.npy", "wb") as fp:
#     #     np.save(fp, z)
#     #
#     # with open("pred_ndps.npy", "wb") as fp:
#     #     np.save(fp, z_pred)
#     # return counter
#
#     return found_ndps


def find_ndps_in_preds(true_pf, pred_pf, i, mdl_hex):
    z, z_pred = np.array(true_pf), np.array(pred_pf)
    assert z.shape[1] == z_pred.shape[1]

    # Defining a data type
    rows, cols = z.shape
    dt_z = {'names': ['f{}'.format(i) for i in range(cols)],
            'formats': cols * [z.dtype]}

    rows, cols = z_pred.shape
    dt_z_pred = {'names': ['f{}'.format(i) for i in range(cols)],
                 'formats': cols * [z_pred.dtype]}

    # Finding intersection
    found_ndps = np.intersect1d(z.view(dt_z), z_pred.view(dt_z_pred))

    return found_ndps


def save_found_ndps(cfg, out_path, i, found_ndps):
    p = out_path / "count_pareto"
    p.mkdir(exist_ok=True, parents=True)
    with open(p / f"found_ndps_{i}.npy", "wb") as fp:
        np.save(fp, found_ndps)


def save_count_pareto(cfg, out_path, i,
                      ndps_in_pred,
                      num_pred_ndps,
                      num_total_ndps,
                      frac_true_ndps_in_pred,
                      frac_ndps_recovered):
    count_pareto_path = out_path / f"count_pareto"
    count_pareto_path.mkdir(exist_ok=True, parents=True)
    count_pareto_file = count_pareto_path / f"{i}.txt"
    count_pareto_file.write_text(f"{i}, {ndps_in_pred}, {num_pred_ndps}, {num_total_ndps}, "
                                 f"{frac_true_ndps_in_pred}, {frac_ndps_recovered}")


def get_prefix(cfg):
    if cfg.deploy.stitching_heuristic == "min_resistance":
        prefix = f"{cfg.deploy.select_all_upto}-mrh{cfg.deploy.lookahead}"
    elif cfg.deploy.stitching_heuristic == "shortest_path":
        prefix = f"{cfg.deploy.select_all_upto}-sph"
    elif cfg.deploy.stitching_heuristic == "mip":
        prefix = f"{cfg.deploy.select_all_upto}-mip"
    else:
        raise ValueError("Invalid heuristic!")

    return prefix


def worker(i, cfg, mdl_hex):
    # Load predicted solution
    out_path = resource_path / f"predictions/xgb/{cfg.prob.name}/{cfg.prob.size}/{cfg.deploy.split}/{mdl_hex}"
    prefix = "" if cfg.deploy.process_connected else get_prefix(cfg)
    prefix = "sols_pred" if cfg.deploy.process_connected else f"{prefix}-sols_pred"
    sols_pred_path = out_path / prefix

    sol_path = sols_pred_path / f"sol_{i}.json"
    if sol_path.exists():
        with open(sol_path, "r") as fp:
            sol_pred = json.load(fp)

        # inst_data = get_instance_data("knapsack", "7_40", cfg.deploy.split, i)
        # order = get_order("knapsack", "MinWt", inst_data)
        # weight = np.array(inst_data["weight"])[order]
        # num_nodes = get_node_count(sol_pred["x"], weight)
        zfp = zipfile.Path(resource_path / f"sols/{cfg.prob.name}/{cfg.prob.size}.zip")
        if zfp.joinpath(f"{cfg.prob.size}/{cfg.deploy.split}/{i}.json").exists():
            zf = zipfile.ZipFile(resource_path / f"sols/{cfg.prob.name}/{cfg.prob.size}.zip")
            with zf.open(f"{cfg.prob.size}/{cfg.deploy.split}/{i}.json") as fp:
                sol = json.load(fp)

            found_ndps = find_ndps_in_preds(sol["z"], sol_pred["z"], i, mdl_hex)
            ndps_in_pred = found_ndps.shape[0]
            num_pred_ndps = len(sol_pred["z"])
            num_total_ndps = len(sol["z"])
            frac_true_ndps_in_pred = ndps_in_pred / num_pred_ndps
            frac_ndps_recovered = ndps_in_pred / num_total_ndps
            print(f"Result for inst {i}: ",
                  ndps_in_pred,
                  frac_true_ndps_in_pred,
                  frac_ndps_recovered)

            save_found_ndps(cfg, sols_pred_path, i, found_ndps)
            save_count_pareto(cfg, sols_pred_path, i,
                              ndps_in_pred,
                              num_pred_ndps,
                              num_total_ndps,
                              frac_true_ndps_in_pred,
                              frac_ndps_recovered)


@hydra.main(version_base="1.2", config_path="./configs", config_name="post_process.yaml")
def main(cfg):
    mdl_name = call_get_model_name(cfg)
    # Convert to hex
    h = hashlib.blake2s(digest_size=32)
    h.update(mdl_name.encode("utf-8"))
    mdl_hex = h.hexdigest()

    # pool = mp.Pool(processes=cfg.nthread)
    # results = [pool.apply_async(worker, args=(i, cfg, mdl_hex)) for i in range(cfg.deploy.from_pid, cfg.deploy.to_pid)]
    # results = [r.get() for r in results]

    worker(cfg.deploy.from_pid, cfg, mdl_hex)


if __name__ == '__main__':
    main()
