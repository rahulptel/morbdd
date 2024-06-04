import numpy as np
import torch

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictorMIS
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problem = "indepset"
split = "val"
n_vars = 100
n_objs = 3
size = f"{n_objs}-{n_vars}"
archive_bdds = path.bdd / f"{problem}/{size}.zip"


def load_model():
    best_model = torch.load(path.resource / "checkpoint/d64-p5-b2-h8-dtk0-dp0.2-t49-v9_1" / "best_model.pt")
    model = ParetoStatePredictorMIS(encoder_type="transformer",
                                    n_node_feat=2,
                                    n_edge_type=2,
                                    d_emb=64,
                                    top_k=5,
                                    n_blocks=2,
                                    n_heads=8,
                                    dropout_token=0,
                                    bias_mha=False,
                                    dropout=0.2,
                                    bias_mlp=False,
                                    h2i_ratio=2,
                                    device=device).to(device)
    model.load_state_dict(best_model["model"])

    return model


def get_node_data(order, bdd):
    data_lst = []

    for lid, layer in enumerate(bdd):
        for nid, node in enumerate(layer):
            # Binary state of the current node
            state = np.zeros(n_vars)
            state[node['s']] = 1
            node_data = np.concatenate(([lid, order[lid + 1]], state))
            data_lst.append(node_data)

    dataset = np.stack(data_lst)

    return dataset


def get_pos_encoding(adj, top_k=5):
    # Calculate position encoding
    U, S, Vh = torch.linalg.svd(adj)
    U = U[:, :top_k]
    S = (torch.diag_embed(S)[:top_k, :top_k]) ** (1 / 2)
    Vh = Vh[:top_k, :]

    L, R = U @ S, S @ Vh
    R = R.permute(1, 0)
    p = torch.cat((L, R), dim=-1)  # n_vars x (2*top_k)

    return p


def fetch_inst_data(pid):
    # Read instance data
    data = get_instance_data(problem, size, split, pid)
    file = f"{size}/{split}/{pid}.json"
    bdd = read_from_zip(archive_bdds, file, format="json")
    # Read order
    order = path.order.joinpath(f"{problem}/{size}/{split}/{pid}.dat").read_text()
    order = np.array(list(map(int, order.strip().split())))
    # Get node data
    obj_coeffs = torch.from_numpy(np.array(data["obj_coeffs"]))
    n_objs, n_vars = obj_coeffs.shape
    obj_id = torch.arange(1, n_objs + 1) / 10
    obj_id = obj_id.repeat((n_vars, 1))
    # obj_id = obj_id.repeat((n_items, 1, 1))
    obj_coeffs = torch.cat((obj_coeffs.T.unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)
    # obj = torch.cat((self.obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)

    adj = torch.from_numpy(np.array(data["adj_list"]))
    pos = get_pos_encoding(adj)
    dataset = get_node_data(order, bdd)
    dataset = torch.from_numpy(dataset)

    return obj_coeffs, adj, pos, dataset


def get_preds(model, obj, adj, pos, dataset):
    n_emb, e_emb = model.token_emb(obj.unsqueeze(0), adj.int().unsqueeze(0), pos.float().unsqueeze(0))
    n_emb, _ = model.encoder(n_emb, e_emb)

    inst_emb = model.graph_encoder(n_emb.sum(1))
    li_emb = model.layer_index_encoder(dataset[:, 0].reshape(-1, 1).float())
    lv_emb = torch.stack([n_emb[0, vid] for vid in dataset[:, 1].int()])
    state_emb = torch.stack([n_emb[0, state].sum(0) for state in dataset[:, 2:].bool()])
    state_emb = model.aggregator(state_emb)
    state_emb += inst_emb + li_emb + lv_emb

    logits = model.predictor(model.ln(state_emb))
    preds = torch.argmax(logits, dim=-1)

    return preds


def get_pareto_states_per_layer(lids, preds):
    pareto_states_per_layer = {}
    prev_lid, counter = int(lids[0]), 0
    for lid, is_pareto in zip(lids[:20], preds[:20]):
        lid = int(lid)

        if prev_lid != lid:
            counter = 0
            prev_lid = lid

        if lid not in pareto_states_per_layer:
            pareto_states_per_layer[lid] = []
        if is_pareto > 0:
            pareto_states_per_layer[lid].append(counter)
        counter += 1

    pareto_states_per_layer_lst = []
    layers = list(pareto_states_per_layer.keys())
    for lid in range(n_vars):
        if lid in layers:
            pareto_states_per_layer_lst.append(pareto_states_per_layer[lid])
        else:
            pareto_states_per_layer_lst.append([])

    return pareto_states_per_layer_lst


def compute_pareto_frontier_on_pareto_bdd(cfg, env, pareto_states, inst_data, order):
    env.set_knapsack_inst(100, 3, inst_data['value'], inst_data['weight'], inst_data['capacity'])
    env.initialize_run(2, 0, 0, -1, [])
    env.compute_pareto_frontier_with_pruning(pareto_states)

    return env


@torch.no_grad()
def main():
    model = load_model()
    model.eval()

    # fetch_inst_data(1000)
    obj, adj, pos, dataset = fetch_inst_data(1000)
    obj, adj, pos, dataset = obj.to(device), adj.to(device), pos.to(device), dataset.to(device)
    preds = get_preds(model, obj, adj, pos, dataset)

    pspl = get_pareto_states_per_layer(dataset[:, 0].tolist(), preds.tolist())


if __name__ == "__main__":
    main()
