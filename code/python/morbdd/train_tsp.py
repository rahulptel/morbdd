import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from morbdd import ResourcePaths as path

# Hyperparams
# 4096 ~ 2.5G
batch_size = 16384
epochs = 10
initial_lr = 1e-2
# Linear warm-up
warmup_steps = 5
# LR Scheduler
factor = 0.9
patience = 5
# AdamW
weight_decay = 1e-3
grad_clip = 1.0


class TSPDataset(Dataset):
    def __init__(self, size, split, device):
        self.size = size
        self.split = split
        self.inst_path = path.inst / f"tsp/{size}/{split}"
        self.dataset_path = path.dataset / f"tsp/{size}/{split}"
        self.dd = json.load(open(path.bdd / f"tsp/{size}/tsp_dd.json", "r"))
        self.D = np.array([]).reshape(-1, 4)

        for p in self.dataset_path.rglob('*.npz'):
            pid = int(p.stem.split('_')[-1])
            d = np.load(p)['arr_0']
            d = np.hstack((np.array([pid] * d.shape[0]).reshape(-1, 1), d))
            self.D = np.vstack((self.D, d))
        self.D = torch.from_numpy(self.D).float().to(device)

        n_items = 1000 if split == "train" else 100
        self.coords = torch.zeros((n_items, 3, 10, 2))
        self.dists = torch.zeros((n_items, 3, 10, 10))
        for p in self.inst_path.rglob("*.npz"):
            pid = int(p.stem.split('_')[-1])
            d = np.load(p)
            self.coords[pid] = torch.from_numpy(d['coords']).float().to(device)
            self.dists[pid] = torch.from_numpy(d['dists']).float().to(device)

    @staticmethod
    def get_layer_weights_exponential(lid):
        return float(np.exp(-0.5 * lid))

    def __len__(self):
        return self.D.shape[0]

    def __getitem__(self, idx):
        pid, lid, nid, score = self.D[idx][0].long(), self.D[idx][1].long(), self.D[idx][2], self.D[idx][3]
        label = 0 if score == 0 else 1
        # d = np.load(self.inst_path / f"tsp_7_{self.size}_{pid}.npz")
        state = self.dd[lid][int(nid.cpu().numpy())]

        return (self.coords[pid],
                self.dists[pid],
                lid,
                torch.tensor(state).float(),
                torch.tensor(self.get_layer_weights_exponential(lid)).float(),
                torch.tensor(score).float(),
                torch.tensor(label).float())


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, bias=True, ln_eps=1e-5, act="relu", dropout=0.0, normalize=False):
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

    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False,
                 is_last_block=False,
                 dropout_attn=0.1,
                 dropout_proj=0.1):
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
        QKV = self.W_qkv(n).reshape(B, -1, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)

        # batch_size x n_heads x n_nodes x d_k
        Q, K, V = QKV[0], QKV[1], QKV[2]

        # Compute edge bias and gate
        # batch_size x n_nodes x n_nodes x n_heads
        E, G = self.W_e(e), F.sigmoid(self.W_g(e))
        # batch_size x n_heads x n_nodes x n_nodes
        E, G = E.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        # batch_size x n_heads x n_nodes
        dynamic_centrality = torch.log(1 + G.sum(-1))

        # Compute implicit attention
        # batch_size x n_heads x n_nodes x n_nodes
        _A_raw = torch.einsum('ijkl,ijlm->ijkm', [Q, K.transpose(-2, -1)])
        _A_raw = _A_raw * (self.d_k ** (-0.5))
        _A_raw = torch.clamp(_A_raw, -5, 5)
        # Add explicit edge bias
        _E = _A_raw + E
        _A = F.softmax(_E, dim=-1)
        # Apply explicit edge gating to V
        # batch_size x n_heads x n_nodes x d_k
        _V = self.drop_attn(_A) @ V
        _V = torch.einsum('ijkl,ijk->ijkl', [_V, dynamic_centrality])
        n = self.drop_proj_n(self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb)))
        e = None if self.O_e is None else self.drop_proj_e(self.O_e(_E.permute(0, 2, 3, 1)))

        return n, e


class GTEncoderLayer(nn.Module):
    def __init__(self,
                 d_emb=32,
                 n_heads=8,
                 bias_mha=False,
                 dropout_attn=0.0,
                 dropout_proj=0.0,
                 bias_mlp=False,
                 dropout_mlp=0.0,
                 h2i_ratio=2,
                 is_last_block=False):
        super(GTEncoderLayer, self).__init__()
        self.is_last_block = is_last_block
        # MHA with edge information
        self.ln_n1 = nn.LayerNorm(d_emb)
        self.ln_e1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttentionWithEdge(d_emb=d_emb,
                                                  n_heads=n_heads,
                                                  bias_mha=bias_mha,
                                                  is_last_block=is_last_block,
                                                  dropout_attn=dropout_attn,
                                                  dropout_proj=dropout_proj)
        # FF
        self.ln_n2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0)
        self.dropout_mlp_n = nn.Dropout(dropout_mlp)

        if not is_last_block:
            # self.dropout_mha_e = nn.Dropout(dropout_mha)
            self.ln_e2 = nn.LayerNorm(d_emb)
            self.mlp_edge = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0)
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
    def __init__(self,
                 d_emb=32,
                 n_layers=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_attn=0.0,
                 dropout_proj=0.0,
                 bias_mlp=False,
                 dropout_mlp=0.0,
                 h2i_ratio=2):
        super(GTEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([GTEncoderLayer(d_emb=d_emb,
                                                            n_heads=n_heads,
                                                            bias_mha=bias_mha,
                                                            dropout_attn=dropout_attn,
                                                            dropout_proj=dropout_proj,
                                                            bias_mlp=bias_mlp,
                                                            dropout_mlp=dropout_mlp,
                                                            h2i_ratio=h2i_ratio,
                                                            is_last_block=i == n_layers - 1)
                                             for i in range(n_layers)])

    def forward(self, n, e):
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n


class TokenEmbedGraph(nn.Module):
    """
    DeepSet-based node and edge embeddings
    """

    def __init__(self, n_node_feat=2, d_emb=32):
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
    def __init__(self,
                 d_emb=32,
                 n_layers=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_attn=0.1,
                 dropout_proj=0.1,
                 bias_mlp=False,
                 dropout_mlp=0.1,
                 h2i_ratio=2):
        super(ParetoNodePredictor, self).__init__()
        self.token_encoder = TokenEmbedGraph(d_emb=d_emb)
        self.graph_encoder = GTEncoder(d_emb=d_emb,
                                       n_layers=n_layers,
                                       n_heads=n_heads,
                                       bias_mha=False,
                                       dropout_attn=0.1,
                                       dropout_proj=0.1,
                                       bias_mlp=False,
                                       dropout_mlp=0.1,
                                       h2i_ratio=2)
        self.state_encoder = nn.Embedding(3, d_emb)
        self.layer_encoder = nn.Sequential(
            nn.Linear(1, d_emb),
            nn.ReLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(d_emb, 2 * d_emb),
            nn.ReLU(),
            nn.Linear(2 * d_emb, 2),
        )

    def forward(self, n, e, l, s):
        n, e = self.token_encoder(n, e)
        n = self.graph_encoder(n, e)  # B x n_vars x d_emb
        B, n_vars, d_emb = n.shape

        l = self.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))  # B x d_emb
        l = l.reshape(B, 1, d_emb)  # B x 1 x d_emb
        last_visit = s[:, -1]
        s_ = s[:, :-1]
        s_[torch.arange(s_.shape[0]), last_visit.long()] = 2
        s_ = self.state_encoder(s_.long())

        dd_node = n + l + s_
        logits = self.node_encoder(dd_node.sum(1))
        return logits


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()

    running_loss = 0.0
    n_items = 0.0
    for i, batch in enumerate(dataloader):
        print(i, len(dataloader))
        batch = [i.to(device, non_blocking=True) for i in batch]
        c, d, l, s, lw, sw, labels = batch

        model.zero_grad()
        logits = model(c, d, l, s)
        loss = loss_fn(logits, labels.long())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.cpu().item() * c.shape[0]
        n_items += c.shape[0]

    epoch_loss = running_loss / n_items
    print("Epoch loss training: ", epoch_loss)

    return epoch_loss


@torch.no_grad()
def test(model, dataloader, loss_fn, device):
    model.eval()
    tn, fp, fn, tp = 0, 0, 0, 0
    running_loss = 0.0
    n_items = 0.0
    for i, batch in enumerate(dataloader):
        print(i, len(dataloader))
        batch = [i.to(device, non_blocking=True) for i in batch]
        c, d, l, s, lw, sw, labels = batch

        model.zero_grad(set_to_none=True)
        logits = model(c, d, l, s)
        loss = loss_fn(logits, labels.long().view(-1))
        running_loss += loss.cpu().item() * c.shape[0]
        n_items += c.shape[0]

        pred_probs = F.softmax(logits.cpu(), dim=-1)
        pred_classes = pred_probs.argmax(dim=-1)
        tn_, fp_, fn_, tp_ = confusion_matrix(labels.cpu().numpy(), pred_classes.cpu().numpy()).ravel()
        tn += tn_
        fp += fp_
        fn += fn_
        tp += tp_

    result = {'loss': running_loss / n_items,
              'tn': tn,
              'fp': fp,
              'fn': fn,
              'tp': tp,
              'precision': tp / (tp + fp),
              'recall': tp / (tp + fn),
              'f1': 2 * tp / ((2 * tp) + fp + fn),
              'accuracy': (tp + tn) / (tp + fn + fp + tn)}
    print(result)
    return result


def is_better(prev_best, new_result, metric):
    if metric == 'f1' or \
            metric == 'accuracy' or \
            metric == 'precision' or \
            metric == 'recall':
        if new_result > prev_best:
            return True

    elif metric == 'loss':
        if new_result < prev_best:
            return True

    return False


def initialize_eval_metric(metric):
    if metric == 'f1' or \
            metric == 'accuracy' or \
            metric == 'precision' or \
            metric == 'recall':
        return 0

    elif metric == 'loss':
        return np.infty


def adjust_learning_rate(epoch, optimizer, scheduler, val_metric):
    """Warmup logic: Gradually increase learning rate."""
    if epoch < warmup_steps:
        lr = initial_lr * (epoch + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step(val_metric)


def training_loop(model,
                  optimizer,
                  loss_fn,
                  train_dataloader,
                  val_dataloader,
                  metric_type,
                  device):
    train_results = []
    val_results = []
    best_metric = initialize_eval_metric(metric_type)
    best_epoch = -1
    val_metric = 0

    # # Set up epoch -1 training performance baseline
    # train_result = test(model, train_dataloader, loss_fn, device)
    # print('Epoch: {}, Split: Train, F1: {}, Recall: {}, Precision: {}'.format(0,
    #                                                                           train_result['f1'],
    #                                                                           train_result['recall'],
    #                                                                           train_result['precision']))

    # Scheduler: Reduce learning rate on plateau (when validation metric stops improving)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=factor)

    for ep in range(epochs):
        adjust_learning_rate(ep, optimizer, scheduler, val_metric)

        train(model,
              train_dataloader,
              loss_fn,
              optimizer,
              device)

        train_result = test(model, train_dataloader, loss_fn, device)
        train_results.append(train_result)
        print('Epoch: {}, Split: Train, F1: {}, Recall: {}, Precision: {}'.format(ep + 1,
                                                                                  train_result['f1'],
                                                                                  train_result['recall'],
                                                                                  train_result['precision']))

        val_result = test(model,
                          val_dataloader,
                          loss_fn,
                          device)
        val_metric = val_result[metric_type]
        val_results.append(val_result)
        print('Epoch: {}, Split: Train, F1: {}, Recall: {}, Precision: {}'.format(ep + 1,
                                                                                  val_result['f1'],
                                                                                  val_result['recall'],
                                                                                  val_result['precision']))
        if is_better(best_metric, val_result[metric_type], metric_type):
            best_metric = val_result[metric_type]
            best_epoch = ep

        print('Best epoch: {}, Best {}: {}'.format(best_epoch,
                                                   metric_type,
                                                   val_results[best_epoch][metric_type]))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Training on :", device)

    # Construct dataset and dataloader
    train_dataset = TSPDataset("3_10", "train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=8,
                                  pin_memory=True,
                                  prefetch_factor=4,
                                  shuffle=True)
    val_dataset = TSPDataset("3_10", "val")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=8,
                                prefetch_factor=4,
                                shuffle=False)

    model = ParetoNodePredictor(d_emb=32,
                                n_layers=2,
                                n_heads=8,
                                bias_mha=False,
                                dropout_attn=0.0,
                                dropout_proj=0.0,
                                bias_mlp=False,
                                dropout_mlp=0.0,
                                h2i_ratio=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    loss_fn = F.cross_entropy
    training_loop(model,
                  opt,
                  loss_fn,
                  train_dataloader,
                  val_dataloader,
                  metric_type='f1',
                  device=device)


if __name__ == '__main__':
    main()
