import json

import torch

from morbdd import ResourcePaths as path
import hydra
import random
from morbdd.utils import read_instance_indepset
import numpy as np
from torch import nn
import torch as T
import torch.nn.functional as F

random.seed(42)


class NodeEncoder(nn.Module):
    def __init__(self, n_features):
        super(NodeEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, 3 * 128)
        self.encX = nn.Linear(3 * 128, 128)
        self.encPN = nn.Linear(3 * 128, 128)
        self.encNN = nn.Linear(3 * 128, 128)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(128 * 3)

    def forward(self, x, pos_adj):
        neg_adj = 1 - pos_adj

        x = self.relu(self.linear1(x))

        x1 = self.encX(x)
        x2 = pos_adj @ self.encPN(x)
        x3 = neg_adj @ self.encNN(x)
        x = torch.cat((x1, x2, x3), dim=2)

        x = self.layer_norm(x)

        return x


class ParetoNodePredictor(nn.Module):
    def __init__(self):
        super(ParetoNodePredictor, self).__init__()
        self.enc1 = nn.Linear(3 * 128, 64)
        self.enc2 = nn.Linear(128, 64)
        self.enc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, ve, se):
        ve = self.relu(self.enc1(ve))
        se = self.relu(self.enc1(se))
        x = self.sigmoid(self.enc2(torch.cat((ve, se), dim=-1)))

        return x


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


def order2rank(order):
    rank = np.zeros_like(order)
    for r, item in enumerate(order):
        rank[item] = r + 1

    rank = rank / len(order)
    return rank


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

    return pos_samples_all, neg_samples_all


def reorder_data(data, order):
    return data


@hydra.main(config_path="configs", config_name="train_mis.yaml", version_base="1.2")
def main(cfg):
    cfg.size = f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    train_ids = list(range(cfg.train.from_pid, cfg.train.to_pid))

    encoder = NodeEncoder(4)
    node_predictor = ParetoNodePredictor()
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

            pos, neg = get_dataset(bdd)

            ranks = []
            for p in pos:
                rank = np.ones_like(order) * -1.0
                for r in range(p[0] + 1):
                    rank[order[r]] = (r + 1) / cfg.prob.n_vars
                ranks.append(rank)
            ranks = np.array(ranks)
            ranks = np.expand_dims(ranks, 2)

            obj_coeffs = np.array(data["obj_coeffs"])
            obj_coeffs = obj_coeffs.T
            obj_coeffs = obj_coeffs[np.newaxis, :, :]
            obj_coeffs = np.repeat(obj_coeffs, 404, 0)

            features = np.concatenate((obj_coeffs, ranks), axis=2)

            features_t = torch.from_numpy(features).float()
            pos_adj = np.expand_dims(data['adj_list'], 0)
            pos_adj = np.repeat(pos_adj, 404, 0)
            pos_adj = torch.from_numpy(pos_adj).float()

            node_emb = encoder(features_t, pos_adj)
            print(node_emb.shape)

            variable_emb = []
            state_emb = []

            for p, ne in zip(pos, node_emb):
                variable_emb.append(ne[order[p[0]]])
                state_emb.append(T.sum(ne[p[1]], dim=0))

            variable_emb = torch.stack(variable_emb, dim=0)
            state_emb = torch.stack(state_emb, dim=0)
            preds = node_predictor(variable_emb, state_emb)

            if i % cfg.val.every == 0:
                pass


if __name__ == "__main__":
    main()
