import io
import json
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset, TensorDataset

from morbdd import ResourcePaths as path


class TSPNodeDataset:
    GRID_DIM = 1000
    MAX_DIST_ON_GRID = ((GRID_DIM**2) + (GRID_DIM**2)) ** (1 / 2)
    INSTS_PER_SPLIT = {"train": 1000, "val": 100, "test": 100}
    PID_OFFSET = {"train": 0, "val": 1000, "test": 1100}
    COORD_DIM = 2
    generator = torch.Generator()
    generator.manual_seed(1337)

    def __init__(
        self,
        n_objs,
        n_vars,
        split,
        device,
        resample,
        subsample,
    ):
        self.n_objs = n_objs
        self.n_vars = n_vars
        self.split = split
        self.device = device
        self.resample = resample
        self.subsample = subsample

        self.size = f"{n_objs}_{n_vars}"
        self.split = split
        self.inst_path = path.inst / f"tsp/{self.size}/{split}"
        self.dataset_path = path.dataset / f"tsp/{self.size}"

        # Send dd to GPU. The nodes in the DD do not change. Only the edge information changes.
        self.dd_flat = None
        # Used to access the dd node from flattened dd. For example, a node in layer lid and an
        # index nid can be obtained as dd_flat_node_idx = self.nodes_in_layer_prefix[lid] + nid
        self.nodes_in_layer_prefix = None
        self.set_dd_flat()
        print("DD flat shape: ", self.dd_flat.shape)

        # Instance data
        self.coords, self.dists = None, None
        self.set_instance_data()
        print("Coords: ", self.coords.shape)
        print("Dists: ", self.dists.shape)

        # Node data
        node_np = np.load(self.dataset_path / f"{split}.npz")["arr_0"]
        self.node_data = torch.from_numpy(node_np).float().to(device)
        n_nodes = self.node_data.shape[0]
        self.ids = torch.arange(n_nodes).to(device)
        self.pos_ids = self.ids[self.node_data[:, -1] == 1]
        self.neg_ids = self.ids[self.node_data[:, -1] != 1]
        # Index 0: negative class weight
        # Index 1: positive class weight
        self.sample_weight = (
            torch.from_numpy(
                np.array(
                    [
                        self.pos_ids.shape[0] / self.node_data.shape[0],
                        self.neg_ids.shape[0] / self.node_data.shape[0],
                    ]
                )
            )
            .float()
            .to(device)
        )

        self.epoch_ids = None
        print("Pos ids: ", self.pos_ids.shape)
        print("Neg ids: ", self.neg_ids.shape)
        self.node_dataset = TensorDataset(
            self.node_data[:, 0:-1], self.node_data[:, -1]
        )
        self.set_epoch_node_ids()

        # Compute layer weights
        # self.lw = self.get_layer_weights_exponential(self.node_data[:, 1]).to(device)

    @staticmethod
    def get_layer_weights_exponential(lid):
        return torch.exp(-0.5 * lid)

    def set_dd_flat(self):
        dd = json.load(open(path.bdd / f"tsp/{self.size}/tsp_dd.json", "r"))
        # Count and prefix
        nodes_in_layer = [len(layer) for layer in dd]
        self.nodes_in_layer_prefix = [0] * len(nodes_in_layer)
        for i in range(1, len(nodes_in_layer)):
            self.nodes_in_layer_prefix[i] = sum(nodes_in_layer[0:i])
        # To-tensor and GPU
        self.nodes_in_layer_prefix = (
            torch.from_numpy(np.array(self.nodes_in_layer_prefix))
            .long()
            .to(self.device)
        )

        self.dd_flat = []
        for lid, layer in enumerate(dd):
            for nid, node in enumerate(layer):
                self.dd_flat.append(node)
        self.dd_flat = torch.from_numpy(np.array(self.dd_flat)).float().to(self.device)

    def set_instance_data(self):
        # Load coordinates and distance matrix to GPU
        n_samples = self.INSTS_PER_SPLIT.get(self.split, None)
        assert n_samples is not None
        self.coords = torch.zeros((n_samples, self.n_objs, self.n_vars, self.COORD_DIM))
        self.dists = torch.zeros((n_samples, self.n_objs, self.n_vars, self.n_vars))
        for p in self.inst_path.rglob("*.npz"):
            pid = int(p.stem.split("_")[-1])
            d = np.load(p)
            idx = pid - self.PID_OFFSET[self.split]
            self.coords[idx] = torch.from_numpy(d["coords"])
            self.dists[idx] = torch.from_numpy(d["dists"])
        self.dists = self.dists.float().to(self.device) / self.MAX_DIST_ON_GRID
        self.coords = self.coords.float().to(self.device) / self.GRID_DIM
        self.coords = torch.cat(
            (self.coords, compute_stat_features(self.dists)), dim=-1
        )

    def get_instance_data(self, pids):
        idxs = pids - self.PID_OFFSET[self.split]
        return self.coords[idxs], self.dists[idxs]

    def set_epoch_node_ids(self):
        pos = self.pos_ids
        # Set neg ids
        neg = self.neg_ids[
            torch.randperm(self.neg_ids.shape[0], generator=self.generator)
        ]
        if self.subsample == 0:
            neg_idx = self.neg_ids.shape[0]
        else:
            neg_idx = self.subsample * self.pos_ids.shape[0]
        neg = neg[:neg_idx]
        # Set epoch ids
        self.epoch_ids = torch.cat((pos, neg))
        self.epoch_ids = self.epoch_ids[
            torch.randperm(self.epoch_ids.shape[0], generator=self.generator)
        ]

    def get_epoch_node_dataset(self):
        if self.resample and self.subsample > 0:
            print(f"Sampling new {self.split} dataset")
            self.set_epoch_node_ids()
            return Subset(self.node_dataset, self.epoch_ids)
        elif not self.resample and self.subsample > 0:
            return Subset(self.node_dataset, self.epoch_ids)
        elif self.subsample == 0:
            return self.node_dataset
        else:
            raise ValueError("Subsample must be greater than 0")

    def __len__(self):
        return len(self.epoch_ids)


class FocalLoss(nn.Module):
    def __init__(self, focal=True, gamma=2):
        super(FocalLoss, self).__init__()
        self.focal = focal
        self.gamma = gamma

    def forward(self, logits, targets, alpha=None):

        probs = torch.sigmoid(logits)
        pt = probs.gather(1, targets.view(-1, 1))
        bce = F.cross_entropy(logits, targets, reduction="none")  # BCELoss
        if self.focal:
            loss = ((1 - pt) ** self.gamma) * bce  # Focal loss
        else:
            loss = bce

        if alpha is not None:
            loss *= alpha  # Class weighted focal loss

        return loss.mean()


class Result:
    def __init__(self):
        self.total_time = None
        self.orig_size = None
        self.restricted_size = None
        self.reduced_size = None
        self.orig_width = None
        self.reduced_width = None
        self.restricted_width = None
        self.cardinality = None
        self.cardinality_raw = None
        self.precision = None
        self.pred_pf = None
        self.n_pred_pf = None
        self.build_time = None
        self.pareto_time = None
        self.edge_agg = None
        self.next_node = None


def get_env(n_objs=3):
    modname = "libtspenvv2o" + str(n_objs)
    libddenv = __import__(modname)
    env = libddenv.TSPEnv()

    return env


def get_instance_data(size, split, pid, seed=7):
    archive = path.inst / f"tsp/{size}.zip"
    inst = f"{size}/{split}/tsp_{seed}_{size}_{pid}.npz"

    # Open the zip file
    with zipfile.ZipFile(archive, "r") as z:
        # Open the .npz file from the zip and load it into numpy
        with z.open(inst) as npz_file:
            # Load the .npz content
            data = np.load(io.BytesIO(npz_file.read()))

    return data


def get_model_str(cfg):
    model_str = [cfg.type, f"v{cfg.version}"]
    if cfg.d_emb != 32:
        model_str.append(f"emb-{cfg.d_emb}")
    if cfg.n_layers != 2:
        model_str.append(f"l-{cfg.n_layers}")
    if cfg.n_heads != 8:
        model_str.append(f"h-{cfg.n_heads}")
    if cfg.act != "relu":
        model_str.append(f"act-{cfg.act}")
    if cfg.concat_emb:
        model_str.append(f"cemb")
    if cfg.dropout_token != 0.0:
        model_str.append(f"dptk-{cfg.dropout_token}")
    if cfg.dropout_attn != 0.0:
        model_str.append(f"dpa-{cfg.dropout_attn}")
    if cfg.dropout_proj != 0.0:
        model_str.append(f"dpp-{cfg.dropout_proj}")
    if cfg.dropout_mlp != 0.0:
        model_str.append(f"dpm-{cfg.dropout_mlp}")
    if cfg.bias_mha:
        model_str.append(f"ba-{cfg.bias_mha}")
    if cfg.bias_mha:
        model_str.append(f"bm-{cfg.bias_mlp}")
    if cfg.h2i_ratio != 2:
        model_str.append(f"h2i-{cfg.h2i_ratio}")

    return model_str


def get_optimizer_str(cfg):
    opt_str = [f"opt-{cfg.type}", f"lr-{cfg.lr}"]
    if cfg.warmup > 0:
        opt_str.append(f"wrm-{cfg.warmup}")
    if cfg.decay_lr:
        opt_str.append(f"dlr")
    if cfg.wd > 0:
        opt_str.append(f"wd-{cfg.wd}")
    if cfg.beta1 != 0.9:
        opt_str.append(f"b1-{cfg.beta1}")
    if cfg.beta2 != 0.999:
        opt_str.append(f"b2-{cfg.beta2}")

    return opt_str


def get_exp_str(cfg):
    exp_str = f"bs-{cfg.batch_size}"
    if cfg.weighted_loss:
        exp_str += f"-wl-"
    exp_str += f"-gcl-{cfg.grad_clip}"
    exp_str += f"-rs-{str(cfg.resample)}"
    exp_str += f"-sst-{str(cfg.subsample.train)}"
    exp_str += f"-ssv-{str(cfg.subsample.val)}"
    return exp_str


def compute_stat_features(dists):
    return torch.cat(
        (
            dists.max(dim=-1, keepdim=True)[0],
            dists.min(dim=-1, keepdim=True)[0],
            dists.std(dim=-1, keepdim=True),
            dists.median(dim=-1, keepdim=True)[0],
            dists.quantile(0.75, dim=-1, keepdim=True)
            - dists.quantile(0.25, dim=-1, keepdim=True),
        ),
        dim=-1,
    )


def get_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    epoch_node_dataset = dataset.get_epoch_node_dataset()
    return DataLoader(
        epoch_node_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
