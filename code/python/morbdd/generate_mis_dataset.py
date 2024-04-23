import hydra
import numpy as np

from morbdd import Const as CONST
from morbdd import ResourcePaths as path
from morbdd.utils import get_instance_data
from morbdd.utils import read_from_zip


def get_size(cfg):
    if cfg.problem_type == 1:
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.problem_type == 2:
        if cfg.graph_type == "stidsen":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
        elif cfg.graph_type == "ba":
            return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def get_lib(bin, n_objs=3):
    libname = 'libbddenv'
    if bin == 'multiobj':
        libname += 'v1'
    elif bin == 'network':
        libname += f'v2o{n_objs}'

    print("Importing lib: ", libname)
    lib = __import__(libname)

    return lib


def get_node_data(bdd, cfg):
    node_data_lst = []
    node_labels_lst = []

    counts = []
    n_total = 0
    for lid, layer in enumerate(bdd):
        neg_data_lst = []
        pos_data_lst = []

        for nid, node in enumerate(layer):
            state = np.zeros(cfg.prob.n_vars)
            state[node['s']] = 1

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

            if node['score'] > 0:
                pos_data_lst.append(node_data)
            else:
                neg_data_lst.append(node_data)

        n_pos = len(pos_data_lst)
        labels = [1] * n_pos
        node_data_lst.extend(pos_data_lst)

        n_neg = min(len(neg_data_lst), int(cfg.pos_to_neg_ratio * n_pos))
        labels += [0] * n_neg
        neg_data_lst = neg_data_lst[:n_neg]
        node_data_lst.extend(neg_data_lst)

        node_labels_lst.extend(labels)

        n_total += n_pos + n_neg
        counts.append((n_pos, n_neg))

    max_feat = np.max([n.shape[0] for n in node_data_lst])
    padded = [np.concatenate((n, np.ones(max_feat - n.shape[0]) * -1)) for n in node_data_lst]

    weights = []
    for nid, n in node_data_lst:
        label = node_labels_lst[nid]
        lid = int(n[0])
        weight = 1 - (counts[lid][1 - label] / n_total)
        weights.append(weight)

    padded = np.stack(padded)
    weights = np.array(weights)
    labels = np.array(node_labels_lst)

    return padded, weights, labels


def worker(rank, cfg):
    archive_bdds = path.bdd / f"{cfg.prob.name}/{cfg.size}.zip"
    for pid in range(cfg.from_pid + rank, cfg.to_pid, cfg.n_processes):
        print("1/10: Fetching instance data and order...")
        data = get_instance_data(cfg.prob.name, cfg.size, cfg.split, pid)

        file = f"{cfg.size}/{cfg.split}/{pid}.json"
        bdd = read_from_zip(archive_bdds, file, format="json")

        order = path.order.joinpath(f"{cfg.prob.name}/{cfg.size}/{cfg.split}/{pid}.dat").read_text()
        order = np.array(map(int, order.strip()))
        print(data.keys(), len(bdd))

        d, w, l = get_node_data(bdd, cfg)
        np.save("d.npz", d)
        np.save("w.npz", w)
        np.save("l.npz", l)


@hydra.main(config_path="configs", config_name="mis_dataset", version_base="1.2")
def main(cfg):
    cfg.size = get_size(cfg)

    worker(0, cfg)
    # pool = mp.Pool(processes=cfg.n_processes)
    # results = []
    #
    # for rank in range(cfg.n_processes):
    #     results.append(pool.apply_async(worker, args=(rank, cfg)))
    #
    # for r in results:
    #     r.get()


if __name__ == "__main__":
    main()
