import json
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictorMIS
from morbdd.train_mis import MISTrainingHelper
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip
from morbdd.utils.mis import get_size

sys.path.append(path.resource / "bin")

RESTRICT = 1
RELAX = 2


class LayerNodeSelector:
    def __init__(self, strategy, width=-1, threshold=0.5):
        self.strategy = strategy
        self.width = width
        self.threshold = threshold

    def __call__(self, lid, scores):

        idx_score = [(i, s) for i, s in enumerate(scores)]
        selection = [0] * len(scores)
        selected_idx, removed_idx = None, None
        if self.strategy == "width":
            if self.width >= len(scores):
                selected_nodes, selected_idx = [1] * len(scores), list(np.arange(len(scores)))
            else:
                idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
                selected_idx = [i[0] for i in idx_score[:self.width]]
                for i in idx_score[:self.width]:
                    selection[i[0]] = 1

        elif self.strategy == "threshold":
            selected_idx = []
            for i in idx_score:
                if i[1] > self.threshold:
                    selection[i[0]] = 1
                    selected_idx.append(i[0])

        removed_idx = list(set(np.arange(len(scores))).difference(set(selected_idx)))

        return selection, selected_idx, removed_idx


class LayerStitcher:
    def __init__(self, strategy="select_all"):
        self.strategy = strategy

    def __call__(self, scores):
        if self.strategy == "select_all":
            return []


def get_env(n_objs=3):
    libbddenv = __import__("libbddenvv2o" + str(n_objs))
    env = libbddenv.BDDEnv()

    return env


def load_model(cfg, model_path):
    best_model = torch.load(path.resource / "checkpoint" / model_path, map_location="cpu")
    model = ParetoStatePredictorMIS(encoder_type="transformer",
                                    n_node_feat=cfg.n_node_feat,
                                    n_edge_type=cfg.n_edge_type,
                                    d_emb=cfg.d_emb,
                                    top_k=cfg.top_k,
                                    n_blocks=cfg.n_blocks,
                                    n_heads=cfg.n_heads,
                                    dropout_token=cfg.dropout_token,
                                    bias_mha=cfg.bias_mha,
                                    dropout=cfg.dropout,
                                    bias_mlp=cfg.bias_mlp,
                                    h2i_ratio=cfg.h2i_ratio)
    model.load_state_dict(best_model["model"])

    return model


def precompute_pos_enc(top_k, adj):
    p = None
    if top_k > 0:
        # Calculate position encoding
        U, S, Vh = torch.linalg.svd(adj)
        U = U[:, :, :top_k]
        S = (torch.diag_embed(S)[:, :top_k, :top_k]) ** (1 / 2)
        Vh = Vh[:, :top_k, :]

        L, R = U @ S, S @ Vh
        R = R.permute(0, 2, 1)
        p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)

    return p


def preprocess_data(data, obj_norm_const=100, max_objs=10, top_k=5):
    # Prepare instance data to be consumed by model
    obj = np.array(data["obj_coeffs"]) / obj_norm_const
    obj = torch.from_numpy(obj).unsqueeze(0)
    obj = append_obj_id(obj, 1, data["n_objs"], data["n_vars"], max_objs)

    adj = torch.from_numpy(np.array(data["adj_list"])).unsqueeze(0)
    pos = precompute_pos_enc(top_k, adj)

    return obj, adj, pos


def append_obj_id(obj, n_items, n_objs, n_vars, max_objs):
    obj_id = torch.arange(1, n_objs + 1) / max_objs
    obj_id = obj_id.repeat((n_vars, 1))
    obj_id = obj_id.repeat((n_items, 1, 1))
    # n_items x n_objs x n_vars x 2
    obj = torch.cat((obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)

    return obj


def get_state_tensor(layer, n_vars):
    states = torch.zeros((len(layer), n_vars))
    for node_id, node in enumerate(layer):
        states[node_id][torch.tensor(node['s']).int()] = 1
    states = states.float()

    return states


@torch.no_grad()
def get_var_embedding(model, obj, adj, pos):
    n_emb, e_emb = model.token_emb(obj.float(), adj.int(), pos.float())
    # Encode: B x n_vars x d_emb
    n_emb = model.node_encoder(n_emb, e_emb)

    return n_emb


@torch.no_grad()
def get_inst_embedding(model, n_emb):
    # B x d_emb
    inst_emb = model.graph_encoder(n_emb.sum(1))
    return inst_emb


@torch.no_grad()
def get_node_scores(model, v_emb, inst_emb, lid, vid, states, threshold=0.5):
    # Layer-index embedding
    # B x d_emb
    n_items, _ = states.shape
    inst_emb = inst_emb.repeat((n_items, 1))

    li_emb = model.layer_index_encoder(torch.tensor(lid).reshape(-1, 1).float())
    li_emb = li_emb.repeat((n_items, 1))

    # Layer-variable embedding
    # B x d_emb
    lv_emb = v_emb[0, vid].unsqueeze(0)
    lv_emb = lv_emb.repeat((n_items, 1))

    # State embedding
    state_emb = torch.einsum("ijk,ij->ik", [v_emb.repeat((n_items, 1, 1)), states])
    state_emb = model.aggregator(state_emb)
    state_emb = state_emb + inst_emb + li_emb + lv_emb

    # Pareto-state predictor
    logits = model.predictor(model.ln(state_emb))

    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def get_run_path(cfg, model_name):
    method = str(cfg.deploy.node_select.prune_from_lid) + "-"
    method += cfg.deploy.node_select.strategy + "-"
    if cfg.deploy.node_select.strategy == "threshold":
        method += str(cfg.deploy.node_select.threshold)

    run_path = path.resource / "sols_pred" / cfg.prob.name / cfg.size / cfg.deploy.split / model_name / method
    return run_path


def load_pf(cfg):
    pid = str(cfg.deploy.pid) + ".json"
    sol_path = path.sol / cfg.prob.name / cfg.size / cfg.deploy.split / pid
    if sol_path.exists():
        print(sol_path)
        with open(sol_path, "r") as fp:
            sol = json.load(fp)
            return sol["z"]

    print("Sol path not found!")


def compute_cardinality(true_pf=None, pred_pf=None):
    z, z_pred = np.array(true_pf), np.array(pred_pf)
    assert z.shape[1] == z_pred.shape[1]

    if z_pred.shape[0] == 0:
        return 0
    else:
        # Defining a data type
        rows, cols = z.shape
        dt_z = {'names': ['f{}'.format(i) for i in range(cols)],
                'formats': cols * [z.dtype]}

        rows, cols = z_pred.shape
        dt_z_pred = {'names': ['f{}'.format(i) for i in range(cols)],
                     'formats': cols * [z_pred.dtype]}

        # Finding intersection
        found_ndps = np.intersect1d(z.view(dt_z), z_pred.view(dt_z_pred))

        return found_ndps.shape[0]


def compute_dd_size(dd):
    s = 0
    for l in dd:
        s += len(l)

    return s


def compute_size_ratio(orig_dd, restricted_dd):
    orig_size = compute_dd_size(orig_dd)
    rest_size = compute_dd_size(restricted_dd)

    return rest_size / orig_size


def load_orig_dd(cfg):
    size = cfg.size + ".zip"
    archive_bdds = path.bdd / cfg.prob.name / size
    file = f"{cfg.size}/{cfg.deploy.split}/{cfg.deploy.pid}.json"
    orig_dd = read_from_zip(archive_bdds, file, format="json")

    return orig_dd


@hydra.main(config_path="configs", config_name="deploy_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    # Load instance data
    data = get_instance_data(cfg.prob.name, cfg.size, cfg.deploy.split, cfg.deploy.pid)
    node_selector = LayerNodeSelector(cfg.deploy.node_select.strategy,
                                      width=cfg.deploy.node_select.width,
                                      threshold=cfg.deploy.node_select.threshold)
    layer_stitcher = LayerStitcher()

    # Load model
    helper = MISTrainingHelper(cfg)
    model_path = helper.get_checkpoint_path() / "best_model.pt"
    model = load_model(cfg, model_path)
    model.eval()

    start = time.time()
    # Preprocess data for ML model
    obj, adj, pos = preprocess_data(data, top_k=cfg.top_k)
    data_preprocess_time = time.time() - start

    # Obtain variable and instance embedding
    # Same for all the nodes of the BDD
    start = time.time()
    v_emb = get_var_embedding(model, obj, adj, pos)
    node_emb_time = time.time() - start

    start = time.time()
    inst_emb = get_inst_embedding(model, v_emb)
    inst_emb_time = time.time() - start

    # Set BDD Manager
    start = time.time()
    order = []
    env = get_env(cfg.prob.n_objs)
    env.reset(cfg.problem_type,
              cfg.preprocess,
              cfg.method,
              cfg.maximization,
              cfg.dominance,
              cfg.bdd_type,
              cfg.maxwidth,
              order)
    env.set_inst(cfg.prob.n_vars, data["n_cons"], cfg.prob.n_objs, data["obj_coeffs"], data["cons_coeffs"], data["rhs"])
    # Initializes BDD with the root node
    env.initialize_dd_constructor()
    # Set the variable used to generate the next layer
    env.set_var_layer(-1)
    lid = 0

    # Restrict and build
    while lid < data["n_vars"] - 1:
        env.generate_next_layer()
        env.set_var_layer(-1)

        layer = env.get_layer(lid + 1)
        states = get_state_tensor(layer, cfg.prob.n_vars)
        vid = torch.tensor(env.get_var_layer()[lid + 1]).int()

        scores = get_node_scores(model, v_emb, inst_emb, lid, vid, states, threshold=0.5)
        lid += 1

        if lid >= cfg.deploy.node_select.prune_from_lid:
            selection, selected_idx, removed_idx = node_selector(lid, scores)
            # Stitch in case of a disconnected BDD
            if len(removed_idx) == len(scores):
                removed_idx = layer_stitcher(scores)
                print("Disconnected at layer: ", {lid})
            # Restrict if necessary
            if len(removed_idx):
                env.approximate_layer(lid, RESTRICT, 1, removed_idx)

    # Generate terminal layer
    env.generate_next_layer()
    build_time = time.time() - start

    start = time.time()
    # Compute pareto frontier
    env.compute_pareto_frontier()
    pareto_time = time.time() - start

    orig_dd = load_orig_dd(cfg)
    restricted_dd = env.get_dd()
    size_ratio = compute_size_ratio(orig_dd, restricted_dd)

    true_pf = load_pf(cfg)
    try:
        pred_pf = env.get_frontier()["z"]
    except:
        pred_pf = None

    cardinality = -1
    if true_pf is not None and pred_pf is not None:
        cardinality = compute_cardinality(true_pf=true_pf, pred_pf=pred_pf)
        cardinality /= len(true_pf)

    run_path = get_run_path(cfg, helper.get_checkpoint_path().stem)
    run_path.mkdir(parents=True, exist_ok=True)

    total_time = data_preprocess_time + node_emb_time + inst_emb_time + build_time + pareto_time
    df = pd.DataFrame([[cfg.size, cfg.deploy.split, cfg.deploy.pid, total_time, size_ratio, cardinality,
                        data_preprocess_time, node_emb_time, inst_emb_time, build_time, pareto_time]],
                      columns=["size", "split", "pid", "total_time", "size", "cardinality", "data_preprocess_time",
                               "node_emb_time", "inst_emb_time", "build_time", "pareto_time"])
    print(df)

    pid = str(cfg.deploy.pid) + ".csv"
    result_path = run_path / pid
    df.to_csv(result_path)


if __name__ == '__main__':
    main()
