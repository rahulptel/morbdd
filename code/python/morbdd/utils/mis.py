import ast
import random

import numpy as np
import torch
from morbdd import ResourcePaths as path
from morbdd.utils import read_from_zip


class CustomCollater:
    def __init__(self, n_vars=100, random_shuffle=True, seed=1231, neg_to_pos_ratio=1):
        self.n_vars = n_vars
        self.random_shuffle = random_shuffle
        self.seed = seed
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def get_states(self, items):
        n_items = len(items)
        states = np.ones((n_items, self.n_vars)) * -1

        for i, indices in enumerate(items):
            states[i][:len(indices)] = indices

        return states

    def __call__(self, batch):
        max_pos = 0

        pids, pids_index, lids, vids, states, labels = [], [], [], [], None, []
        for item in batch:
            item["json"] = ast.literal_eval(item["json"].decode("utf-8"))

            n_pos = len(item["json"]["pos"])
            n_neg = min(len(item["json"]["neg"]), self.neg_to_pos_ratio * n_pos)

            if item["json"]["pid"] not in pids:
                pids.append(item["json"]["pid"])
            index = pids.index(item["json"]["pid"])
            pids_index.extend([index] * (n_pos + n_neg))
            lids.extend([item["json"]["lid"]] * (n_pos + n_neg))
            vids.extend([item["json"]["vid"]] * (n_pos + n_neg))
            labels.extend([1] * n_pos)
            labels.extend([0] * n_neg)

            states_pos = self.get_states(item["json"]["pos"])
            states = states_pos if states is None else np.vstack((states, states_pos))

            # For validation set keep the shuffling fixed
            if not self.random_shuffle:
                random.seed(self.seed)
            random.shuffle(item["json"]["neg"])

            states_neg = self.get_states(item["json"]["neg"][:n_neg])
            states = states_neg if states is None else np.vstack((states, states_neg))

        pids, pids_index, lids, vids = (torch.tensor(pids).int(), torch.tensor(pids_index).int(),
                                        torch.tensor(lids).float(), torch.tensor(vids).int())
        indices = torch.from_numpy(states).int()
        labels = torch.tensor(labels).long()

        return pids, pids_index, lids, vids, indices, labels


def get_size(cfg):
    if cfg.graph_type == "stidsen":
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}"
    elif cfg.graph_type == "ba":
        return f"{cfg.prob.n_objs}-{cfg.prob.n_vars}-{cfg.prob.attach}"


def read_instance(archive, inst):
    if inst.split(".")[-1] == "npz":
        data = read_from_zip(archive, inst, format="npz")
    else:
        raw_data = read_from_zip(archive, inst)

        data = {"obj_coeffs": [], "cons_coeffs": [], "rhs": []}

        data["n_vars"], data["n_cons"] = list(map(int, raw_data.readline().strip().split()))
        data["n_objs"] = int(raw_data.readline())
        data["adj_list"] = np.zeros((data["n_vars"], data["n_vars"]))
        data["adj_list_comp"] = np.ones((data["n_vars"], data["n_vars"]))
        for i in range(data["n_vars"]):
            data["adj_list"][i, i] = 1
            data["adj_list_comp"][i, i] = 0

        for _ in range(data["n_objs"]):
            data["obj_coeffs"].append(list(map(int, raw_data.readline().split())))
        # print(data["obj_coeffs"])

        for _ in range(data["n_cons"]):
            n_vars_per_con = list(map(int, raw_data.readline().strip().split()))[0]
            non_zero_vars = list(map(int, raw_data.readline().strip().split()))
            # print(n_vars_per_con, non_zero_vars)
            non_zero_vars = [i - 1 for i in non_zero_vars]
            data["cons_coeffs"].append(non_zero_vars)

            for i in range(len(non_zero_vars)):
                i_var = non_zero_vars[i]
                for j in range(i + 1, len(non_zero_vars)):
                    j_var = non_zero_vars[j]
                    data["adj_list"][i_var, j_var] = 1
                    data["adj_list"][j_var, i_var] = 1
                    data["adj_list_comp"][i_var, j_var] = 0
                    data["adj_list_comp"][j_var, i_var] = 0

    return data


def get_instance_data(size, split, pid):
    prefix = "ind_7"
    archive = path.inst / f"indepset/{size}.zip"
    suffix = "npz" if len(size.split("-")) > 2 else "dat"

    inst = f'{size}/{split}/{prefix}_{size}_{pid}.{suffix}'
    data = read_instance(archive, inst)

    return data


def get_checkpoint_path(cfg):
    exp = ""
    if cfg.model == "transformer":
        exp = ("d{}-p{}-b{}-h{}-dtk{}"
               "-dp{}-t{}-v{}").format(cfg.d_emb,
                                       cfg.top_k,
                                       cfg.n_blocks,
                                       cfg.n_heads,
                                       cfg.dropout_token,
                                       cfg.dropout,
                                       cfg.dataset.shard.train.end,
                                       cfg.dataset.shard.val.end)
    ckpt_path = path.resource / "checkpoint" / exp

    return ckpt_path
