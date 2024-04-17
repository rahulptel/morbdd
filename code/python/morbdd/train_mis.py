import json

from morbdd import ResourcePaths as path
import hydra
import random
from morbdd.utils import read_instance_indepset
import numpy as np

random.seed(42)


def pad_samples(samples, max_parents, n_vars):
    for sample in samples:
        if len(sample[2]) < max_parents:
            sample[2] = np.vstack((sample[2],
                                   np.zeros((max_parents - len(sample[2]), n_vars))
                                   ))

        if len(sample[3]) < max_parents:
            sample[3] = np.vstack((sample[3],
                                   np.zeros((max_parents - len(sample[3]), n_vars))
                                   ))

    return samples


def get_dataset(bdd):
    dataset = []
    pos_samples_all = []
    neg_samples_all = []
    max_parents = 0

    for lid, layer in enumerate(bdd):
        pos_samples = []
        neg_samples = []
        for nid, node in enumerate(layer):
            # s, op, zp, score, pareto
            state = np.zeros(len(bdd) + 1)
            state[node['s']] = 1

            zp_states = []
            for zp in node['zp']:
                if lid > 0:
                    zp_state = np.zeros(len(bdd) + 1)
                    zp_state[bdd[lid - 1][zp]['s']] = 1
                else:
                    zp_state = np.ones(len(bdd) + 1)
                zp_states.append(zp_state)

            if len(zp_states) > max_parents:
                max_parents = len(zp_states)
            zp_states = np.array(zp_states).reshape(-1, len(bdd) + 1)

            op_states = []
            for op in node['op']:
                if lid > 0:
                    op_state = np.zeros(len(bdd) + 1)
                    op_state[bdd[lid - 1][op]['s']] = 1
                else:
                    op_state = np.ones(len(bdd) + 1)
                op_states.append(op_state)

            if len(op_states) > max_parents:
                max_parents = len(op_states)
            op_states = np.array(op_states).reshape(-1, len(bdd) + 1)

            node_data = [lid, state, zp_states, op_states, node['score']]
            if node['pareto']:
                pos_samples.append(node_data)
            else:
                neg_samples.append(node_data)

        if len(neg_samples) > len(pos_samples):
            random.shuffle(neg_samples)
            neg_samples = neg_samples[:len(pos_samples)]

        pos_samples_all.extend(pos_samples)
        neg_samples_all.extend(neg_samples)

    pos_samples_all = pad_samples(pos_samples_all, max_parents, len(bdd) + 1)
    neg_samples_all = pad_samples(neg_samples_all, max_parents, len(bdd) + 1)

    return pos_samples_all


def reorder_data(data, order):
    return data


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    train_ids = list(range(cfg.train.from_pid, cfg.train.to_pid))

    for epoch in range(cfg.train.epochs):
        random.shuffle(train_ids)
        for i, train_id in enumerate(train_ids):
            archive = path.inst / cfg.prob.name / f"{cfg.size}.zip"
            file = f"{cfg.size}/train/ind_7_{cfg.size}_{train_id}.dat"
            data = read_instance_indepset(archive, file)

            order_path = path.order / cfg.prob.name / cfg.size / "train" / f"{train_id}.dat"
            print()

            order = list(map(int, order_path.read_text(encoding="utf-8").strip().split(" ")))

            bdd_path = path.bdd / cfg.prob.name / cfg.size / "train" / f"{train_id}.json"
            bdd = json.load(open(bdd_path, "r"))

            dataset = get_dataset(bdd)

            if i % cfg.val.every == 0:
                pass


if __name__ == "__main__":
    main()
