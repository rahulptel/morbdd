import multiprocessing as mp
import random

import hydra
import numpy as np
import torch

from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip
from morbdd.utils import get_layer_weights


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def get_node_data(cfg, bdd):
    data_lst = []
    labels_lst = []
    counts = []
    n_total = 0
    for lid, layer in enumerate(bdd):
        neg_data_lst = []
        pos_data_lst = []
        for nid, node in enumerate(layer):
            # Binary state of the current node
            state = np.zeros(cfg.prob.n_vars)
            state[node['s']] = 1
            # Binary state of the parent state
            if cfg.with_parent:
                n_parents_op, n_parents_zp = 0, 0
                zp_lst, op_lst = np.array([]), np.array([])
                if lid > 0:
                    n_parents_zp = len(node['zp'])
                    n_parents_op = len(node['op'])

                    for zp in node['zp']:
                        zp_state = np.zeros(cfg.prob.n_vars)
                        zp_state[bdd[lid - 1][zp]['s']] = 1
                        zp_lst = np.concatenate((zp_lst, zp_state))

                    op_lst = np.array([])
                    for op in node['op']:
                        op_state = np.zeros(cfg.prob.n_vars)
                        op_state[bdd[lid - 1][op]['s']] = 1
                        op_lst = np.concatenate((op_lst, op_state))

                node_data = np.concatenate(([lid],
                                            state,
                                            [n_parents_zp],
                                            zp_lst,
                                            [n_parents_op],
                                            op_lst))
            else:
                node_data = np.concatenate(([lid],
                                            state))

            if node['score'] > 0:
                pos_data_lst.append(node_data)
            else:
                neg_data_lst.append(node_data)

        # Get label
        n_pos = len(pos_data_lst)
        labels = [1] * n_pos
        data_lst.extend(pos_data_lst)
        # Undersample negative class
        n_neg = min(len(neg_data_lst), int(cfg.neg_to_pos_ratio * n_pos))
        labels += [0] * n_neg
        random.shuffle(neg_data_lst)
        neg_data_lst = neg_data_lst[:n_neg]
        data_lst.extend(neg_data_lst)
        labels_lst.extend(labels)

        # Update class counts
        n_total += n_pos + n_neg
        counts.append((n_neg, n_pos))

    # Pad features
    max_feat = np.max([n.shape[0] for n in data_lst])
    padded = [np.concatenate((n, np.ones(max_feat - n.shape[0]) * -1)) for n in data_lst]

    # Sample weights
    layer_weights = get_layer_weights(True, cfg.layer_weight, cfg.prob.n_vars)
    weights = []
    for nid, n in enumerate(data_lst):
        label, lid = labels_lst[nid], int(n[0])
        weight = 1 - (counts[lid][label] / n_total)
        weight *= layer_weights[lid]
        weights.append(weight)

    padded, weights, labels = np.stack(padded), np.array(weights), np.array(labels_lst)
    padded = np.hstack((weights.reshape(-1, 1), padded))

    return padded, labels


def worker(rank, cfg):
    archive_bdds = path.bdd / f"{cfg.prob.name}/{cfg.size}.zip"
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.n_processes):
        print("1/10: Fetching instance data and order...")
        # Read instance data
        data = get_instance_data(cfg.prob.name, cfg.size, cfg.split, pid)
        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive_bdds, file, format="json")
        # Read order
        order = path.order.joinpath(f"{cfg.prob.name}/{cfg.size}/{cfg.split}/{pid}.dat").read_text()
        order = np.array(list(map(int, order.strip().split())))
        # Get node data
        obj_coeffs = torch.from_numpy(np.array(data["obj_coeffs"]))
        adj = torch.from_numpy(np.array(data["adj_list"]))
        X, Y = get_node_data(cfg, bdd)
        X, Y, order = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(order)

        # Save data
        file_path = path.dataset / f"{cfg.prob.name}/{cfg.size}/{cfg.split}"
        prefix = f"{cfg.layer_weight}-{cfg.neg_to_pos_ratio}"
        file_path /= f"{prefix}-parent" if cfg.with_parent else f"{prefix}-no-parent"
        file_path.mkdir(exist_ok=True, parents=True)
        obj = {"x": X, "y": Y, "order": order, "obj_coeffs": obj_coeffs, "adj": adj}
        torch.save(obj, file_path / f"{pid}.pt")


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


if __name__ == "__main__":
    main()
