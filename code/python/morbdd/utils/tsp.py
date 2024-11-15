import io
import zipfile

import numpy as np

from morbdd import ResourcePaths as path

import torch


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
    model_str = f"{cfg.model.type}-v{cfg.model.version}-"
    if cfg.model.d_emb != 32:
        model_str += f"-emb-{cfg.model.d_emb}"
    if cfg.model.n_layers != 2:
        model_str += f"-l-{cfg.model.n_layers}"
    if cfg.model.n_heads != 8:
        model_str += f"-h-{cfg.model.n_heads}"
    if cfg.model.dropout_token != 0.0:
        model_str += f"-dptk-{cfg.model.dropout_token}"
    if cfg.model.dropout_attn != 0.0:
        model_str += f"-dpa-{cfg.model.dropout_attn}"
    if cfg.model.dropout_proj != 0.0:
        model_str += f"-dpp-{cfg.model.dropout_proj}"
    if cfg.model.dropout_mlp != 0.0:
        model_str += f"-dpm-{cfg.model.dropout_mlp}"
    if cfg.model.bias_mha:
        model_str += f"-ba-{cfg.model.bias_mha}"
    if cfg.model.bias_mha:
        model_str += f"-bm-{cfg.model.bias_mlp}"
    if cfg.model.h2i_ratio != 2:
        model_str += f"-h2i-{cfg.model.h2i_ratio}"

    return model_str


def get_optimizer_str(cfg):
    opt_str = "opt"
    opt_str += f"-{cfg.optimizer}"
    # if cfg.weight_decay != 1e-3:
    #     opt_str += f"-wd{cfg.weight_decay}"
    # opt_str += f"-lr-{cfg.max_lr}-{cfg.warmup_steps}"
    opt_str += f"-lr-{cfg.max_lr}"
    # if cfg.decay is not None and cfg.decay != "Cosine":
    #     opt_str += f"-{cfg.decay}"
    # if cfg.batch_size != 512:
    opt_str += f"-bs-{cfg.batch_size}"
    if cfg.grad_clip != 1.0:
        opt_str += f"-gcl-{cfg.grad_clip}"
    opt_str += f"-rs-{cfg.resample}"
    opt_str += f"-ss-{cfg.subsample}"
    return opt_str


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
