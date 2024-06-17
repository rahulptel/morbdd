import ast
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
from torch.utils.data import Subset

from morbdd import ResourcePaths as path
from morbdd.utils import TrainingHelper
from morbdd.utils import read_from_zip


class MISTrainingHelper(TrainingHelper):
    def __init__(self, cfg):
        super(MISTrainingHelper, self).__init__()
        self.cfg = cfg

    def get_dataset(self, split, from_pid, to_pid):
        bdd_node_dataset = np.load(str(path.dataset) + f"/{self.cfg.prob.name}/{self.cfg.size}/{split}.npy")
        valid_rows = (from_pid <= bdd_node_dataset[:, 0])
        valid_rows &= (bdd_node_dataset[:, 0] < to_pid)

        bdd_node_dataset = bdd_node_dataset[valid_rows]
        if split == "val":
            bdd_node_dataset[:, 0] -= 1000
        if split == "test":
            bdd_node_dataset[:, 0] -= 1100

        obj, adj = [], []
        for pid in range(from_pid, to_pid):
            data = get_instance_data(self.cfg.size, split, pid)
            obj.append(data["obj_coeffs"])
            adj.append(data["adj_list"])
        obj, adj = np.array(obj), np.stack(adj)
        dataset = MISBDDNodeDataset(bdd_node_dataset, obj, adj, top_k=self.cfg.top_k)

        return dataset

    @staticmethod
    def get_train_dataset(cfg, dataset):
        frac_per_epoch = cfg.dataset.train.frac_per_epoch
        if frac_per_epoch == 1:
            train_dataset = dataset
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            train_dataset = Subset(dataset, indices[: int(len(dataset) * frac_per_epoch)])

        return train_dataset

    def get_checkpoint_path(self):
        exp = self.get_exp_name()
        ckpt_path = path.resource / "checkpoint" / exp

        return ckpt_path

    def get_exp_name(self):
        exp = self.get_model_str() + self.get_opt_str() + self.get_dataset_str()
        if self.cfg.with_timestamp:
            exp += "-" + str(float(time.time()))

        return exp

    def get_model_str(self):
        model_str = ""
        if self.cfg.encoder_type == "transformer":
            model_str += "tf"
            if self.cfg.d_emb != 64:
                model_str += f"-emb-{self.cfg.d_emb}"
            if self.cfg.top_k != 5:
                model_str += f"-k-{self.cfg.top_k}"
            if self.cfg.n_blocks != 2:
                model_str += f"-l-{self.cfg.n_blocks}"
            if self.cfg.n_heads != 8:
                model_str += f"-h-{self.cfg.n_heads}"
            if self.cfg.dropout_token != 0.0:
                model_str += f"-dptk-{self.cfg.dropout_token}"
            if self.cfg.dropout != 0.2:
                model_str += f"-dp-{self.cfg.dropout}"
            if self.cfg.bias_mha:
                model_str += f"-ba-{self.cfg.bias_mha}"
            if self.cfg.bias_mha:
                model_str += f"-bm-{self.cfg.bias_mlp}"
            if self.cfg.h2i_ratio != 2:
                model_str += f"-h2i-{self.cfg.h2i_ratio}"

        elif self.cfg.encoder_type == "gat":
            model_str += "gat"
            if self.cfg.d_emb != 64:
                model_str += f"-emb-{self.cfg.d_emb}"

        return model_str

    def get_opt_str(self):
        ostr = "-"
        if self.cfg.opt.name != "Adam":
            ostr += self.cfg.opt.name
        if self.cfg.opt.lr != 1e-3:
            ostr += f"-lr-{self.cfg.opt.lr}"
        if self.cfg.opt.wd != 1e-4:
            ostr += f"-wd-{self.cfg.opt.wd}"
        if self.cfg.clip_grad != 1.0:
            ostr += f"-clip-{self.cfg.clip_grad}"
        if self.cfg.norm_type != 2.0:
            ostr += f"-norm-{self.cfg.norm_type}"

        if ostr == "-":
            ostr = ""

        return ostr

    def get_dataset_str(self):
        dstr = "-"
        if not self.cfg.validate_on_master:
            dstr += "nvm"

        dstr = "-t"
        if self.cfg.dataset.train.from_pid != 0:
            dstr += f"-f-{self.cfg.dataset.train.from_pid}"
        if self.cfg.dataset.train.to_pid != 1000:
            dstr += f"-t-{self.cfg.dataset.train.to_pid}"
        if self.cfg.dataset.validate_on == "val":
            dstr += f"-v"
            if self.cfg.dataset.val.from_pid != 1000:
                dstr += f"-f-{self.cfg.dataset.val.from_pid}"
            if self.cfg.dataset.val.to_pid != 1100:
                dstr += f"-t-{self.cfg.dataset.val.to_pid}"
        else:
            dstr += "-t"

        if dstr == "-":
            dstr = ""

        return dstr


class MISBDDNodeDataset(Dataset):
    def __init__(self, bdd_node_dataset, obj, adj, top_k=5, norm_const=100, max_objs=10):
        super(MISBDDNodeDataset, self).__init__()
        self.norm_const = norm_const
        self.max_objs = max_objs

        self.nodes = torch.from_numpy(bdd_node_dataset.astype('int16'))
        perm = torch.randperm(self.nodes.shape[0])
        self.nodes = self.nodes[perm]

        self.top_k = top_k
        self.obj, self.adj = torch.from_numpy(obj) / self.norm_const, torch.from_numpy(adj)
        self.append_obj_id()
        self.pos = self.precompute_pos_enc(top_k, self.adj)

    def __getitem__(self, item):
        pid = self.nodes[item, 0]

        return (self.obj[pid],
                self.adj[pid],
                self.pos[pid] if self.top_k > 0 else None,
                self.nodes[item, 0],
                self.nodes[item, 1],
                self.nodes[item, 2],
                self.nodes[item, 3:103],
                self.nodes[item, 103])

    def __len__(self):
        return self.nodes.shape[0]

    @staticmethod
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

    def append_obj_id(self):
        n_items, n_objs, n_vars = self.obj.shape
        obj_id = torch.arange(1, n_objs + 1) / self.max_objs
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((n_items, 1, 1))
        # n_items x n_objs x n_vars x 2
        self.obj = torch.cat((self.obj.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)


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


def stitch(bdd, lidx, method="parent", lookahead=1):
    if method == "parent":
        pass
    elif method == "min_resistance":
        pass
    elif method == "mip":
        pass

    return bdd
