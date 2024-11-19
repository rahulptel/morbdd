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
    model_str = f"{cfg.type}-v{cfg.version}-"
    if cfg.d_emb != 32:
        model_str += f"-emb-{cfg.d_emb}"
    if cfg.n_layers != 2:
        model_str += f"-l-{cfg.n_layers}"
    if cfg.n_heads != 8:
        model_str += f"-h-{cfg.n_heads}"
    if cfg.act != "relu":
        model_str += f"-act-{cfg.act}"
    if cfg.concat_emb:
        model_str += f"-cemb-"
    if cfg.dropout_token != 0.0:
        model_str += f"-dptk-{cfg.dropout_token}"
    if cfg.dropout_attn != 0.0:
        model_str += f"-dpa-{cfg.dropout_attn}"
    if cfg.dropout_proj != 0.0:
        model_str += f"-dpp-{cfg.dropout_proj}"
    if cfg.dropout_mlp != 0.0:
        model_str += f"-dpm-{cfg.dropout_mlp}"
    if cfg.bias_mha:
        model_str += f"-ba-{cfg.bias_mha}"
    if cfg.bias_mha:
        model_str += f"-bm-{cfg.bias_mlp}"
    if cfg.h2i_ratio != 2:
        model_str += f"-h2i-{cfg.h2i_ratio}"

    return model_str


def get_optimizer_str(cfg):
    opt_str = f"opt-{cfg.type}-lr-{cfg.lr}"
    if cfg.warmup > 0:
        opt_str += f"-wrm-{cfg.warmup}"
    if cfg.decay_lr:
        opt_str += f"-dlr"
    if cfg.wd > 0:
        opt_str += f"-wd-{cfg.wd}"
    if cfg.beta1 != 0.9:
        opt_str += f"-b1-{cfg.beta1}"
    if cfg.beta2 != 0.999:
        opt_str += f"-b2-{cfg.beta2}"

    return opt_str


def get_exp_str(cfg):
    exp_str = f"bs-{cfg.batch_size}"
    if cfg.weighted_loss:
        exp_str = f"-wl-"
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
