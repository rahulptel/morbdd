import json
import signal
import time

import hydra
import numpy as np
import pandas as pd
import torch

from morbdd import ResourcePaths as path
from morbdd.utils import MetricCalculator
from morbdd.utils import handle_timeout
from morbdd.utils.tsp import Result, compute_stat_features
from morbdd.utils.tsp import get_env
from morbdd.utils.tsp import get_instance_data
from .train_tsp import ParetoNodePredictor

RESTRICT = 1
GRID_DIM = 1000
MAX_DIST_ON_GRID = ((GRID_DIM**2) + (GRID_DIM**2)) ** (1 / 2)


def save_result(cfg, result, seed, pid):
    df = pd.DataFrame(
        [
            [
                cfg.prob.size,
                cfg.split,
                pid,
                result.total_time,
                result.orig_size,
                result.restricted_size,
                result.reduced_size,
                result.orig_width,
                result.restricted_width,
                result.reduced_width,
                result.cardinality,
                result.cardinality_raw,
                result.precision,
                len(result.pred_pf),
                result.build_time,
                result.pareto_time,
            ]
        ],
        columns=[
            "size",
            "split",
            "pid",
            "total_time",
            "orig_size",
            "rest_size",
            "reduced_size",
            "orig_width",
            "rest_width",
            "reduced_width",
            "cardinality",
            "cardinality_raw",
            "pred_precision",
            "n_pred_pf",
            "build_time",
            "pareto_time",
        ],
    )

    pid = str(pid) + f"_{seed}.csv"
    save_path = (
        path.resource
        / "sols_pred"
        / cfg.prob.name
        / cfg.prob.size
        / cfg.split
        / cfg.model.type
        / str(cfg.max_width)
    )
    # save_path = save_path / f"{exp_path}" / str(cfg.baseline.max_width)
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / pid
    print(df)
    df.to_csv(save_path)


@torch.no_grad()
def get_node_scores(layer_id, node_emb, model, layer, n_vars=10):
    layer = torch.from_numpy(np.array(layer)).float()
    B = layer.shape[0]
    node_emb = node_emb.repeat(B, 1, 1)

    last_visit = layer[:, -1]
    visit_mask = layer[:, :-1]
    visit_mask[torch.arange(B), last_visit.long()] = 2
    visit_enc = model.visit_encoder(visit_mask.long())
    # B x d_emb
    node_visit = model.node_visit_encoder2(
        model.node_visit_encoder1((node_emb + visit_enc)).sum(1)
    )
    customer_enc = node_emb[torch.arange(B), last_visit.long()]
    l = torch.tensor(np.array([layer_id] * B))
    l_enc = model.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))

    preds = model.pareto_predictor(node_visit + customer_enc + l_enc)
    preds = torch.softmax(preds, dim=-1)
    preds = preds.cpu().numpy()
    return preds[:, -1]


@torch.no_grad()
def build_dd(env, max_width, inst, model):
    coords = (torch.from_numpy(inst["coords"]) / GRID_DIM).float().unsqueeze(0)
    dists = (torch.from_numpy(inst["dists"]) / MAX_DIST_ON_GRID).float().unsqueeze(0)
    node_feat = torch.cat((coords, compute_stat_features(dists)), dim=-1)
    node_emb, edge_emb = model.token_encoder(
        node_feat,
        dists,
    )
    node_emb = model.graph_encoder(node_emb, edge_emb)  # B x n_vars x d_emb

    # Restrict and build
    lid = 2
    while True:
        print("Building layer: ", lid)
        is_done = env.generate_next_layer()
        layer = env.get_layer(lid)
        print("Size: ", len(layer))
        if len(layer) > max_width:
            # Sort nodes in descending order of scores and remove the last ones
            scores = get_node_scores(
                lid - 1, node_emb, model, layer, n_vars=node_emb.shape[0]
            )
            idx_scores = [(i, s) for i, s in enumerate(scores)]
            idx_scores.sort(key=lambda x: x[1], reverse=True)
            nodes_to_remove = [i for i, _ in idx_scores[max_width:]]
            # nodes_to_remove.sort()
            # print(nodes_to_remove)
            env.approximate_layer(lid, RESTRICT, nodes_to_remove)

        lid += 1
        if is_done:
            break


def run_pipeline(
    cfg, inst, max_width, model, metric_calculator, result, exact_dd=None, true_pf=None
):
    print("Run pipeline")
    signal.signal(signal.SIGALRM, handle_timeout)
    env = get_env(n_objs=cfg.prob.n_objs)

    env.reset()
    env.set_inst(
        cfg.prob.n_vars,
        cfg.prob.n_objs,
        inst["dists"].astype(int).tolist(),
    )
    env.initialize_dd_constructor()
    start = time.time()
    build_dd(env, max_width, inst, model)
    result.build_time = time.time() - start

    # Compute pareto frontier
    try:
        signal.alarm(1800)
        start = time.time()
        env.compute_pareto_frontier()
        result.pareto_time = time.time() - start
        result.pred_pf = env.get_frontier()["z"]
    except:
        result.pred_pf = None
        result.pareto_time = 1800
    signal.alarm(0)
    result.total_time = result.build_time + result.pareto_time
    if result.pred_pf is not None:
        restricted_dd = env.get_dd()
        layer_sizes = [len(layer) for layer in restricted_dd]
        result.restricted_size = sum(layer_sizes)
        result.restricted_width = max(layer_sizes)

    if true_pf is not None:
        res = metric_calculator.compute_cardinality(true_pf, result.pred_pf)
        result.cardinality_raw = res["cardinality_raw"]
        result.cardinality = res["cardinality"]
        result.precision = res["precision"]

    return result


@hydra.main(config_path="./configs", config_name="deploy_tsp.yaml", version_base="1.2")
def main(cfg):
    exact_dd = json.load(
        open(path.bdd / f"{cfg.prob.name}/{cfg.prob.size}/tsp_dd.json", "r")
    )
    layer_sizes = [len(layer) for layer in exact_dd]
    max_width = int(max(layer_sizes) * (cfg.max_width / 100))
    print("Size of the exact dd: ", sum(layer_sizes))
    print("Width of the exact dd: ", max(layer_sizes))
    print("Max width of the restricted dd: ", max_width)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Deploying on :", device)
    model = ParetoNodePredictor(cfg.model).to(device)
    model.eval()
    print(
        f"Checkpoint path: {path.checkpoint}/{cfg.prob.prefix}/{cfg.prob.size}/{cfg.model.type}_best_model.pt"
    )
    ckpt = torch.load(
        f"{path.checkpoint}/{cfg.prob.prefix}/{cfg.prob.size}/{cfg.model.type}_best_model.pt",
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    metric_calculator = MetricCalculator(cfg.prob.n_vars)
    for pid in range(cfg.from_pid, cfg.to_pid):
        result = Result()

        # Load instance
        inst = get_instance_data(cfg.prob.size, cfg.split, pid, seed=7)
        # Load true PF
        true_pf = None
        sol_path = (
            path.sol / cfg.prob.name / cfg.prob.size / cfg.split / f"sol_{pid}.npz"
        )
        if not sol_path.exists():
            continue

        true_pf = np.load(sol_path)
        true_pf = true_pf["z"]
        # Build dd layer-by-layer with pruning using node information
        result = run_pipeline(
            cfg,
            inst,
            max_width,
            model,
            metric_calculator,
            result,
            exact_dd=exact_dd,
            true_pf=true_pf,
        )
        save_result(cfg, result, 7, pid)


if __name__ == "__main__":
    main()
