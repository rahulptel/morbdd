import json
import pickle as pkl

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, TensorDataset
import time
from morbdd import ResourcePaths as path
from morbdd.utils.tsp import get_model_str, get_optimizer_str


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

    def compute_stat_features(self):
        return torch.cat(
            (
                self.dists.max(dim=-1, keepdim=True)[0],
                self.dists.min(dim=-1, keepdim=True)[0],
                self.dists.std(dim=-1, keepdim=True),
                self.dists.median(dim=-1, keepdim=True)[0],
                self.dists.quantile(0.75, dim=-1, keepdim=True)
                - self.dists.quantile(0.25, dim=-1, keepdim=True),
            ),
            dim=-1,
        )

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
        self.coords = torch.cat((self.coords, self.compute_stat_features()), dim=-1)

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


class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_hid,
        d_out,
        bias=True,
        ln_eps=1e-5,
        act="relu",
        dropout=0.0,
        normalize=False,
    ):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.normalize = normalize
        if self.normalize:
            self.ln = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hid, bias=bias)
        self.linear2 = nn.Linear(d_hid, d_out, bias=bias)
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x) if self.normalize else x
        x = self.act(self.linear1(x))
        x = self.dropout(x) if self.dropout.p > 0 else x
        x = self.act(self.linear2(x))

        return x


class MultiHeadSelfAttentionWithEdge(nn.Module):
    """
    Based on: Global Self-Attention as a Replacement for Graph Convolution
    https://arxiv.org/pdf/2108.03348
    """

    def __init__(
        self,
        d_emb=64,
        n_heads=8,
        bias_mha=False,
        is_last_block=False,
        dropout_attn=0.1,
        dropout_proj=0.1,
    ):
        super(MultiHeadSelfAttentionWithEdge, self).__init__()
        assert d_emb % n_heads == 0

        self.d_emb = d_emb
        self.d_k = d_emb // n_heads
        self.n_heads = n_heads
        self.is_last_block = is_last_block
        self.drop_attn = nn.Dropout(dropout_attn)
        self.drop_proj_n = nn.Dropout(dropout_proj)

        # Node Q, K, V params
        self.W_qkv = nn.Linear(d_emb, 3 * d_emb, bias=bias_mha)
        self.O_n = nn.Linear(n_heads * self.d_k, d_emb, bias=bias_mha)

        # Edge bias and gating parameters
        self.W_g = nn.Linear(d_emb, n_heads, bias=bias_mha)
        self.W_e = nn.Linear(d_emb, n_heads, bias=bias_mha)

        # Output mapping params
        if is_last_block:
            self.O_e = None
        else:
            self.O_e = nn.Linear(n_heads, d_emb, bias=bias_mha)
            self.drop_proj_e = nn.Dropout(dropout_proj)

    def forward(self, n, e):
        """
        n : batch_size x n_nodes x d_emb
        e : batch_size x n_nodes x n_nodes x d_emb
        """
        assert e is not None
        B = n.shape[0]

        # Compute QKV and reshape
        # 3 x batch_size x n_heads x n_nodes x d_k
        QKV = (
            self.W_qkv(n)
            .reshape(B, -1, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )

        # batch_size x n_heads x n_nodes x d_k
        Q, K, V = QKV[0], QKV[1], QKV[2]

        # Compute edge bias and gate
        # batch_size x n_nodes x n_nodes x n_heads
        E, G = self.W_e(e), torch.sigmoid(self.W_g(e))
        # batch_size x n_heads x n_nodes x n_nodes
        E, G = E.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        # batch_size x n_heads x n_nodes
        dynamic_centrality = torch.log(1 + G.sum(-1))

        # Compute implicit attention
        # batch_size x n_heads x n_nodes x n_nodes
        _A_raw = torch.einsum("ijkl,ijlm->ijkm", [Q, K.transpose(-2, -1)])
        _A_raw = _A_raw * (self.d_k ** (-0.5))
        _A_raw = torch.clamp(_A_raw, -5, 5)
        # Add explicit edge bias
        _E = _A_raw + E
        _A = torch.softmax(_E, dim=-1)
        # Apply explicit edge gating to V
        # batch_size x n_heads x n_nodes x d_k
        _V = self.drop_attn(_A) @ V
        _V = torch.einsum("ijkl,ijk->ijkl", [_V, dynamic_centrality])
        n = self.drop_proj_n(self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb)))
        e = (
            None
            if self.O_e is None
            else self.drop_proj_e(self.O_e(_E.permute(0, 2, 3, 1)))
        )

        return n, e


class GTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_emb=32,
        n_heads=8,
        bias_mha=False,
        dropout_attn=0.0,
        dropout_proj=0.0,
        bias_mlp=False,
        dropout_mlp=0.0,
        h2i_ratio=2,
        is_last_block=False,
    ):
        super(GTEncoderLayer, self).__init__()
        self.is_last_block = is_last_block
        # MHA with edge information
        self.ln_n1 = nn.LayerNorm(d_emb)
        self.ln_e1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttentionWithEdge(
            d_emb=d_emb,
            n_heads=n_heads,
            bias_mha=bias_mha,
            is_last_block=is_last_block,
            dropout_attn=dropout_attn,
            dropout_proj=dropout_proj,
        )
        # FF
        self.ln_n2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(
            d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0
        )
        self.dropout_mlp_n = nn.Dropout(dropout_mlp)

        if not is_last_block:
            # self.dropout_mha_e = nn.Dropout(dropout_mha)
            self.ln_e2 = nn.LayerNorm(d_emb)
            self.mlp_edge = MLP(
                d_emb,
                h2i_ratio * d_emb,
                d_emb,
                bias=bias_mlp,
                normalize=False,
                dropout=0.0,
            )
            self.dropout_mlp_e = nn.Dropout(dropout_mlp)

    def forward(self, n, e):
        n_norm = self.ln_n1(n)
        e_norm = self.ln_e1(e)
        n_, e_ = self.mha(n_norm, e_norm)
        n = n + n_

        n = n + self.dropout_mlp_n(self.mlp_node(self.ln_n2(n)))
        if not self.is_last_block:
            e = e + e_
            e = e + self.dropout_mlp_e(self.mlp_edge(self.ln_e2(e)))

        return n, e


class GTEncoder(nn.Module):
    def __init__(
        self,
        d_emb=32,
        n_layers=2,
        n_heads=8,
        bias_mha=False,
        dropout_attn=0.0,
        dropout_proj=0.0,
        bias_mlp=False,
        dropout_mlp=0.0,
        h2i_ratio=2,
    ):
        super(GTEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                GTEncoderLayer(
                    d_emb=d_emb,
                    n_heads=n_heads,
                    bias_mha=bias_mha,
                    dropout_attn=dropout_attn,
                    dropout_proj=dropout_proj,
                    bias_mlp=bias_mlp,
                    dropout_mlp=dropout_mlp,
                    h2i_ratio=h2i_ratio,
                    is_last_block=i == n_layers - 1,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, n, e):
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n


class TokenEmbedGraph(nn.Module):
    """
    DeepSet-based node and edge embeddings
    """

    def __init__(self, n_node_feat=7, d_emb=32):
        super(TokenEmbedGraph, self).__init__()
        self.linear1 = nn.Linear(n_node_feat, 2 * d_emb)
        self.linear2 = nn.Linear(2 * d_emb, d_emb)
        self.linear3 = nn.Linear(1, d_emb)
        self.linear4 = nn.Linear(d_emb, d_emb)

    def forward(self, n, e):
        n = F.relu(self.linear1(n))  # B x n_objs x n_vars x (2 * d_emb)
        n = n.sum(1)  # B x n_vars x (2 * d_emb)
        n = F.relu(self.linear2(n))  # B x n_vars x d_emb

        e = e.unsqueeze(-1)
        e = F.relu(self.linear3(e))  # B x n_objs x n_vars x n_vars x d_emb
        e = e.sum(1)  # B x n_vars x n_vars x d_emb
        e = F.relu(self.linear4(e))  # B x n_vars x n_vars x d_emb

        return n, e


class ParetoNodePredictor(nn.Module):
    # NOT_VISITED = 0
    # VISITED = 1
    # LAST_VISITED = 2
    NODE_VISIT_TYPES = 3
    N_LAYER_INDEX = 1
    N_CLASSES = 2

    def __init__(
        self,
        d_emb=32,
        n_layers=2,
        n_heads=8,
        bias_mha=False,
        dropout_attn=0.0,
        dropout_proj=0.0,
        bias_mlp=False,
        dropout_mlp=0.0,
        h2i_ratio=2,
    ):
        super(ParetoNodePredictor, self).__init__()
        self.token_encoder = TokenEmbedGraph(d_emb=d_emb)
        self.graph_encoder = GTEncoder(
            d_emb=d_emb,
            n_layers=n_layers,
            n_heads=n_heads,
            bias_mha=bias_mha,
            dropout_attn=dropout_attn,
            dropout_proj=dropout_proj,
            bias_mlp=bias_mlp,
            dropout_mlp=dropout_mlp,
            h2i_ratio=h2i_ratio,
        )
        self.visit_encoder = nn.Embedding(self.NODE_VISIT_TYPES, d_emb)
        self.node_visit_encoder1 = nn.Sequential(
            nn.Linear(d_emb, h2i_ratio * d_emb),
            nn.ReLU(),
        )
        self.node_visit_encoder2 = nn.Sequential(
            nn.Linear(h2i_ratio * d_emb, d_emb),
            nn.ReLU(),
        )
        self.layer_encoder = nn.Sequential(
            nn.Linear(self.N_LAYER_INDEX, d_emb),
            nn.ReLU(),
        )
        self.pareto_predictor = nn.Sequential(
            nn.Linear(d_emb, h2i_ratio * d_emb),
            nn.ReLU(),
            nn.Linear(h2i_ratio * d_emb, self.N_CLASSES),
        )

    def forward(self, n, e, l, s):
        n, e = self.token_encoder(n, e)
        n = self.graph_encoder(n, e)  # B x n_vars x d_emb
        B, n_vars, d_emb = n.shape

        last_visit = s[:, -1]
        visit_mask = s[:, :-1]
        visit_mask[torch.arange(B), last_visit.long()] = 2
        visit_enc = self.visit_encoder(visit_mask.long())

        # B x d_emb
        node_visit = self.node_visit_encoder2(
            self.node_visit_encoder1((n + visit_enc)).sum(1)
        )
        customer_enc = n[torch.arange(B), last_visit.long()]
        l_enc = self.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))

        return self.pareto_predictor(node_visit + customer_enc + l_enc)


def save_model(save_path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def save_result(save_path, ep, global_step, train_result, val_result):
    pkl.dump(
        {
            "epoch": ep,
            "global_step": global_step,
            "train_result": train_result,
            "val_result": val_result,
        },
        save_path,
    )


def flatten_batch(batch, dataset):
    node_feat, label = batch

    pid, lid, nid, ns = (
        node_feat[:, 0].long(),
        node_feat[:, 1].long(),
        node_feat[:, 2].long(),
        node_feat[:, 3],
    )
    # DD node data
    flat_idx = dataset.nodes_in_layer_prefix[lid] + nid
    dd_node_data = dataset.dd_flat[flat_idx]
    # Instance data
    coords, dists = dataset.get_instance_data(pid)
    # Layer weight
    lw = dataset.get_layer_weights_exponential(lid)
    label = label.long()

    return coords, dists, lid, dd_node_data, lw, ns, label


@torch.no_grad()
def test(cfg, model, dataset, dataloader, loss_fn):
    model.eval()
    tn, fp, fn, tp = 0, 0, 0, 0
    running_loss = 0.0
    n_items = 0.0
    for i, batch in enumerate(dataloader):
        # coords, dists, lids, states, lw, sw, labels = batch
        coords, dists, lids, states, lw, sw, labels = flatten_batch(batch, dataset)

        logits = model(coords, dists, lids, states)
        loss = loss_fn(logits, labels, reduction="none")
        if cfg.weighted_loss:
            loss *= lw + sw  # Add layer weight and pareto-score weights
        loss = loss.mean()
        running_loss += loss.cpu().item() * coords.shape[0]
        n_items += coords.shape[0]

        pred_probs = F.softmax(logits.cpu(), dim=-1)
        pred_classes = pred_probs.argmax(dim=-1)
        tn_, fp_, fn_, tp_ = confusion_matrix(
            labels.cpu().numpy(), pred_classes.cpu().numpy()
        ).ravel()
        tn += tn_
        fp += fp_
        fn += fn_
        tp += tp_

    result = {
        "loss": running_loss / n_items,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / (tp + fn + fp + tn),
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    if tp + fp > 0:
        result["precision"] = tp / (tp + fp)

    if tp + fn > 0:
        result["recall"] = tp / (tp + fn)

    if result["precision"] > 0 and result["recall"] > 0:
        result["f1"] = (2 * tp) / ((2 * tp) + fp + fn)

    return result


def is_better(prev_best, new_result, metric):
    if (
        metric == "f1"
        or metric == "accuracy"
        or metric == "precision"
        or metric == "recall"
    ):
        if new_result > prev_best:
            return True

    elif metric == "loss":
        if new_result < prev_best:
            return True

    return False


def initialize_eval_metric(metric):
    if (
        metric == "f1"
        or metric == "accuracy"
        or metric == "precision"
        or metric == "recall"
    ):
        return 0

    elif metric == "loss":
        return np.infty


def adjust_learning_rate(step, optimizer, scheduler, val_metric, max_lr, warmup_steps):
    """Linearly increase learning rate and then decrease the learning rate using ReduceLROnPlateau scheduler."""
    if step < warmup_steps:
        lr = max_lr * (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    # else:
    #     scheduler.step(val_metric)


def print_eval_result(split, ep, max_epochs, global_step, max_steps, result):
    print(
        "Epoch {}/{}, Step {}/{}, Split: {}".format(
            ep, max_epochs, global_step, max_steps, split
        )
    )
    print(
        "\tF1: {}, Recall: {}, Precision: {}, Loss: {}".format(
            result["f1"], result["recall"], result["precision"], result["loss"]
        )
    )


def training_loop(
    cfg, model, optimizer, loss_fn, train_dataset, val_dataset, metric_type
):
    exp_str = get_model_str(cfg) + "-" + get_optimizer_str(cfg)
    exp_path = path.checkpoint / "tsp" / cfg.prob.size / exp_str
    exp_path.mkdir(exist_ok=True, parents=True)
    print("Saving experiment results to :", str(exp_path))

    # Scheduler: Reduce learning rate on plateau (when validation metric stops improving)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=factor)
    max_steps = (len(train_dataset) // cfg.batch_size) * cfg.epochs
    # warmup_steps = int((warmup_steps_percent / 100) * max_steps)
    # print('Training epochs: {}, max steps: {}, warm-up steps: {}'.format(epochs, max_steps, warmup_steps))

    times = {"train": 0}
    train_results, val_results = [], []
    global_step, val_metric, best_epoch, best_step = 0, 0, -1, -1
    best_metric = initialize_eval_metric(metric_type)
    val_node_dataset = val_dataset.get_epoch_node_dataset()
    val_dataloader = DataLoader(
        val_node_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    tick = time.time()
    for ep in range(cfg.epochs):
        train_epoch_node_dataset = train_dataset.get_epoch_node_dataset()
        print("Train dataset: ", len(train_epoch_node_dataset))
        train_dataloader = DataLoader(
            train_epoch_node_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print("Train dataloader: ", len(train_dataloader))
        for i, batch in enumerate(train_dataloader):
            model.train()
            global_step += 1
            # adjust_learning_rate(global_step, optimizer, scheduler, val_metric, warmup_steps)
            coords, dists, lids, states, lw, sw, labels = flatten_batch(
                batch, train_dataset
            )
            logits = model(coords, dists, lids, states)
            loss = loss_fn(logits, labels, reduction="none")
            if cfg.weighted_loss:
                loss *= lw + sw  # Add layer weight and pareto-score weights
            loss = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if (global_step + 1) % cfg.eval_every == 0:
                train_result = test(
                    cfg, model, train_dataset, train_dataloader, loss_fn
                )
                train_result.update({"epoch": ep, "global_step": global_step})
                train_results.append(train_result)
                print_eval_result(
                    "Train", ep, cfg.epochs, global_step, max_steps, train_result
                )

                val_result = test(cfg, model, val_dataset, val_dataloader, loss_fn)
                val_metric = val_result[metric_type]
                val_result.update({"epoch": ep, "global_step": global_step})
                val_results.append(val_result)
                print_eval_result(
                    "Val", ep, cfg.epochs, global_step, max_steps, val_result
                )

                save_path = exp_path / f"model_{ep}_{global_step}.pt"
                save_model(save_path, model, optimizer)
                save_path = open(str(exp_path / f"result_{ep}_{global_step}.pkl"), "wb")
                save_result(save_path, ep, global_step, train_result, val_result)

                if is_better(best_metric, val_result[metric_type], metric_type):
                    best_metric = val_result[metric_type]
                    best_epoch = ep
                    best_step = global_step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f"{exp_path}/best_model.pt",
                    )

                print(
                    "\tBest epoch:step={}:{}, Best {}: {}".format(
                        best_epoch, best_step, metric_type, best_metric
                    )
                )
    times["train"] = time.time() - tick
    print("Wallclock time: ", times["train"] / 3600)
    pkl.dump(times, open(str(exp_path / "log.pkl"), "wb"))


@hydra.main(config_path="./configs", config_name="train_tsp.yaml", version_base="1.2")
def main(cfg):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on :", device)

    # Construct dataset
    train_dataset = TSPNodeDataset(
        cfg.prob.n_objs,
        cfg.prob.n_vars,
        "train",
        device,
        resample=cfg.resample,
        subsample=cfg.subsample,
    )
    print(
        "Train dataset: ",
        len(train_dataset),
        "Resample: ",
        cfg.resample,
        "Subsample: ",
        cfg.subsample,
    )
    val_dataset = TSPNodeDataset(
        cfg.prob.n_objs,
        cfg.prob.n_vars,
        "val",
        device,
        resample=False,
        subsample=0,
    )
    print("Val dataset: ", len(val_dataset))

    model = ParetoNodePredictor(
        d_emb=cfg.model.d_emb,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        bias_mha=cfg.model.bias_mha,
        dropout_attn=cfg.model.dropout_attn,
        dropout_proj=cfg.model.dropout_proj,
        bias_mlp=cfg.model.bias_mlp,
        dropout_mlp=cfg.model.dropout_mlp,
        h2i_ratio=cfg.model.h2i_ratio,
    ).to(device)
    optimizer_cls = getattr(torch.optim, cfg.optimizer)
    opt = optimizer_cls(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )

    loss_fn = F.cross_entropy
    training_loop(
        cfg, model, opt, loss_fn, train_dataset, val_dataset, metric_type="f1"
    )


if __name__ == "__main__":
    main()
