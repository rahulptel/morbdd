import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from morbdd import ResourcePaths as path
from morbdd.model import ParetoStatePredictorMIS
from morbdd.train_mis import MISTrainingHelper
from morbdd.utils import get_device
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip
from morbdd.utils.mis import get_instance_data
from morbdd.utils.mis import get_size
from morbdd.train_mis import validate


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


def get_stats_per_layer(pspl, preds):
    pass


def get_approx_pareto_frontier(preds):
    pass


def get_node_data(n_vars, order, bdd):
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


def fetch_inst_data(problem, size, split, pid, archive_bdds):
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
    dataset = get_node_data(n_vars, order, bdd)
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


def get_pareto_states_per_layer(n_vars, lids, preds):
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


def learning_eval(cfg, model, device, helper, pin_memory=False, multi_gpu=False):
    print("Evaluating learning metrics on validation set")
    val_dataset = helper.get_dataset("val", cfg.dataset.val.from_pid, cfg.dataset.val.to_pid)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.n_worker_dataloader)

    stats = validate(val_dataloader, model, device, helper, pin_memory=pin_memory, multi_gpu=multi_gpu)
    meta_stats = helper.compute_meta_stats(stats)
    helper.print_stats("val", meta_stats)


def downstream_eval(cfg, model, device, helper, pin_memory=False, multi_gpu=False):
    from_pid, to_pid = cfg.dataset[cfg.eval.split].from_pid, cfg.dataset[cfg.eval.split].to_pid
    for pid in range(from_pid, to_pid):
        obj, adj, pos, dataset = fetch_inst_data(cfg.prob.name, cfg.size, cfg.eval.split, pid,
                                                 path.bdd / f"{cfg.prob.name}/{cfg.prob.size}.zip")
        obj, adj, pos, dataset = obj.to(device), adj.to(device), pos.to(device), dataset.to(device)

        preds = get_preds(model, obj, adj, pos, dataset)
        pspl = get_pareto_states_per_layer(cfg.pron.n_vars, dataset[:, 0].tolist(), preds.tolist())
        get_stats_per_layer(pspl, preds)
        apx_frontier, apx_stats = get_approx_pareto_frontier(preds)


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)
    helper = MISTrainingHelper(cfg)

    # Set-up device
    device, device_str, pin_memory, master, device_id = get_device(multi_gpu=False)
    print("Device :", device)

    # Load model for eval
    ckpt_path = helper.get_checkpoint_path()
    print("Checkpoint path: {}".format(ckpt_path))
    model = load_model(cfg, ckpt_path / "best_model.pt")
    model.to(device)
    model.eval()

    if cfg.eval.task == "learning":
        assert cfg.eval.split != "test"
        learning_eval(cfg, model, device, helper, pin_memory=pin_memory, multi_gpu=False)
    if cfg.eval.task == "downstream":
        downstream_eval(cfg, model, device, helper, pin_memory=pin_memory, multi_gpu=False)


if __name__ == "__main__":
    main()
