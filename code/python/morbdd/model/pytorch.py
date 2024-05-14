import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch_geometric.nn import GATConv
# from torch_geometric.data import Data


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, bias=True, ln_eps=1e-5, act="relu",
                 dropout=0.1, normalize=False):
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
        if self.normalize:
            x = self.ln1(x)
        x = self.dropout(self.act(self.linear1(x)))
        x = self.act(self.linear2(x))

        return x


class KnapsackInstanceTokenizer:
    def __init__(self, max_objs=10, device=None):
        self.max_objs = max_objs
        self.device = device

    def tokenize(self, n):
        objs = n[:, :, :-2]
        weight = n[:, :, -2:]

        B, n_objs, n_vars = objs.shape
        obj_id = torch.arange(1, n_objs + 1) / self.max_objs
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((B, 1, 1)).to(self.device)
        o = torch.cat((objs.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)

        return o, weight


class TokenEmbedKnapsack(nn.Module):
    def __init__(self, max_objs=10, n_obj_feat=2, n_con_feat=2, d_emb=64, device=None):
        super(TokenEmbedKnapsack, self).__init__()
        self.max_objs = max_objs
        self.device = device
        self.linear1 = nn.Linear(n_obj_feat, 2 * d_emb)
        self.linear2 = nn.Linear(2 * d_emb, d_emb)
        self.mlp = MLP(n_con_feat, 2 * d_emb, d_emb)

    def forward(self, o, c):
        # batch_size x n_objs x n_vars x 2 * d_emb
        o = F.relu(self.linear1(o))
        # batch_size x n_vars x 2 * d_emb
        o = o.sum(1)
        # batch_size x n_vars x d_emb
        o = F.relu(self.linear2(o))
        # batch_size x n_vars x d_emb
        c = self.mlp(c)
        # variable features
        n = o + c

        return n


class MISInstanceTokenizer:
    def __init__(self, top_k=0, max_objs=10, device=None):
        self.top_k = top_k
        self.max_objs = max_objs
        self.device = device

    def tokenize(self, n, e=None):
        p = None  # position features
        if self.top_k and e is not None:
            # Calculate position encoding
            U, S, Vh = torch.linalg.svd(e)
            U = U[:, :, :self.top_k]
            S = (torch.diag_embed(S)[:, :self.top_k, :self.top_k]) ** (1 / 2)
            Vh = Vh[:, :self.top_k, :]

            L, R = U @ S, S @ Vh
            R = R.permute(0, 2, 1)
            p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)

        B, n_objs, n_vars = n.shape
        obj_id = torch.arange(1, n_objs + 1) / self.max_objs
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((B, 1, 1)).to(self.device)
        # B x n_objs x n_vars x 2
        n = torch.cat((n.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)

        return n, p


class TokenEmbedGraph(nn.Module):
    """
    Tokenize graph input to obtain position-aware node embedding and
    edge embeddings
    """

    def __init__(self, n_node_feat, n_edge_type=2, d_emb=64, top_k=0):
        super(TokenEmbedGraph, self).__init__()
        self.top_k = top_k
        self.linear1 = nn.Linear(n_node_feat, 2 * d_emb)
        self.linear2 = nn.Linear(2 * d_emb, d_emb)
        self.edge_encoder = nn.Embedding(n_edge_type, d_emb)
        if self.top_k > 0:
            self.pos_encoder = nn.Linear(top_k * 2, d_emb)

    def forward(self, n, e, p):
        # Calculate node and edge encodings
        n = F.relu(self.linear1(n))  # B x n_vars x n_objs x 2 * d_emb
        # Sum aggregate objectives
        n = n.sum(2)  # B x n_vars x 2 * d_emb
        n = F.relu(self.linear2(n))  # B x n_vars x d_emb
        # Update node encoding with positional encoding based on SVD
        if self.top_k:
            p = self.pos_encoder(p)  # B x n_vars x d_emb
            n = n + p

        e = self.edge_encoder(e)  # B x n_vars x n_vars x d_emb

        return n, e


class MultiHeadSelfAttention(nn.Module):
    """
    Based on: Global Self-Attention as a Replacement for Graph Convolution
    https://arxiv.org/pdf/2108.03348
    """

    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False,
                 with_edge=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_k = d_emb // n_heads
        self.d_emb = d_emb
        self.n_heads = n_heads
        # Node Q, K, V params
        self.W_q = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_k = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_v = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.O_n = nn.Linear(n_heads * self.d_k, d_emb, bias=bias_mha)
        if with_edge:
            # Edge bias and gating parameters
            self.W_g = nn.Linear(d_emb, n_heads, bias=bias_mha)
            self.W_e = nn.Linear(d_emb, n_heads, bias=bias_mha)
            # Output mapping params
            self.O_e = nn.Linear(n_heads, d_emb, bias=bias_mha)

        self.dynamic_forward = getattr(self, "forward_with_e" if with_edge is None else "forward_without_e")

    def forward(self, n, e=None):
        """
        n : batch_size x n_nodes x d_emb
        e : batch_size x n_nodes x n_nodes x d_emb
        """
        B = n.shape[0]

        # Compute QKV and reshape
        # batch_size x n_nodes x (n_heads * d_k)
        Q, K, V = self.W_q(n), self.W_k(n), self.W_v(n)
        # batch_size x n_nodes x n_heads x d_k
        Q = Q.view(B, -1, self.n_heads, self.d_k)
        K = K.view(B, -1, self.n_heads, self.d_k)
        V = V.view(B, -1, self.n_heads, self.d_k)
        # batch_size x n_heads x n_nodes x d_k
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        n, e = self.dynamic_forward(B, Q, K, V, e=e)

        return n, e

    def forward_without_e(self, B, Q, K, V, e=None):
        """Normal attention"""

        # Compute implicit attention
        # batch_size x n_heads x n_nodes x n_nodes
        _A = torch.einsum('ijkl,ijlm->ijkm', [Q, K.transpose(-2, -1)])
        _A = _A * ((self.d_k) ** (-0.5))
        _A = torch.clamp(_A, -5, 5)
        _A = F.softmax(_A, dim=-1)

        # batch_size x n_heads x n_nodes x d_k
        _V = _A @ V
        # batch_size x n_nodes x d_emb
        n = self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb))

        return n, e

    def forward_with_e(self, B, Q, K, V, e=None):
        """Attention conditioned on edge features"""

        # Compute edge bias and gate
        # batch_size x n_nodes x n_nodes x n_heads
        E, G = self.W_e(e), F.sigmoid(self.W_g(e))
        # batch_size x n_heads x n_nodes x n_nodes
        E, G = E.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        # batch_size x n_heads x n_nodes
        dynamic_centrality = torch.log(1 + G.sum(-1))

        # Compute implicit attention
        # batch_size x n_heads x n_nodes x n_nodes
        _A = torch.einsum('ijkl,ijlm->ijkm', [Q, K.transpose(-2, -1)])
        _A = _A * ((self.d_k) ** (-0.5))
        _A = torch.clamp(_A, -5, 5)
        # Add explicit edge bias
        _E = _A + E
        _A = F.softmax(_E, dim=-1)
        # Apply explicit edge gating to V
        # batch_size x n_heads x n_nodes x d_k
        _V = _A @ V
        _V = torch.einsum('ijkl,ijk->ijkl', [_V, dynamic_centrality])

        n = self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb))
        e = self.O_e(_E.permute(0, 2, 3, 1))

        return n, e


class GTEncoderLayer(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 with_edge=False):
        super(GTEncoderLayer, self).__init__()

        self.ln_n1 = nn.LayerNorm(d_emb)
        self.ln_e1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttention(d_emb=d_emb,
                                          n_heads=n_heads,
                                          bias_mha=bias_mha,
                                          with_edge=with_edge)
        self.dropout_mha = nn.Dropout(dropout_mha)

        self.ln_n2 = nn.LayerNorm(d_emb)
        self.ln_e2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp)
        self.mlp_edge = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp)
        self.dropout_mlp = nn.Dropout(dropout_mlp)

    def forward(self, n, e=None):
        n, e = self.ln_n1(n), e if e is None else self.ln_e1(e)
        n_, e_ = self.mha(n, e)
        n, e = n + self.dropout_mha(n_), e if e is None else e + self.dropout_mha(e_)

        n, e = self.ln_n2(n), e if e is None else self.ln_e2(e)
        n, e = n + self.dropout_mlp(self.mlp_node(n)), e if e is None else e + self.dropout_mlp(self.mlp_edge(e))

        return n, e


#
# class GATEncoder(nn.Module):
#     def __init__(self, n_emb, n_heads=8, n_blocks=2, dropout=0.1):
#         super(GATEncoder, self).__init__()
#         self.conv1 = GATConv(n_emb, n_heads, heads=n_heads, dropout=dropout)
#         self.conv2 = GATConv(8 * n_heads, 8, heads=1, concat=False, dropout=dropout)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


class GTEncoder(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 with_edge=False):
        super(GTEncoder, self).__init__()
        self.encoder_blocks = clones(GTEncoderLayer(d_emb=d_emb,
                                                    n_heads=n_heads,
                                                    bias_mha=bias_mha,
                                                    dropout_mha=dropout_mha,
                                                    bias_mlp=bias_mlp,
                                                    dropout_mlp=dropout_mlp,
                                                    h2i_ratio=h2i_ratio,
                                                    with_edge=with_edge),
                                     n_blocks)

    def forward(self, n, e=None):
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n, e


encoder_factory = {
    "transformer": GTEncoder
}


def get_encoder(encoder_type,
                d_emb=64,
                n_blocks=2,
                n_heads=8,
                bias_mha=False,
                dropout_mha=0,
                bias_mlp=True,
                dropout_mlp=0.1,
                h2i_ratio=2,
                with_edge=False,
                dropout_gat=0.1):
    if encoder_type == "transformer":
        return GTEncoder(d_emb=d_emb,
                         n_blocks=n_blocks,
                         n_heads=n_heads,
                         bias_mha=bias_mha,
                         dropout_mha=dropout_mha,
                         bias_mlp=bias_mlp,
                         dropout_mlp=dropout_mlp,
                         h2i_ratio=h2i_ratio,
                         with_edge=with_edge)
    # if self.encoder_type == "gat":
    #     return encoder_cls(d_emb,
    #                        n_heads=n_heads,
    #                        n_blocks=n_blocks,
    #                        dropout=dropout_gat)


class ParetoStatePredictorMIS(nn.Module):
    def __init__(self,
                 encoder_type="transformer",
                 n_node_feat=2,
                 n_edge_type=2,
                 d_emb=64,
                 top_k=5,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 device=None):
        super(ParetoStatePredictorMIS, self).__init__()
        self.device = device
        self.tokenizer = MISInstanceTokenizer(top_k=top_k, device=device)
        self.token_emb = TokenEmbedGraph(n_node_feat,
                                         n_edge_type=n_edge_type,
                                         d_emb=d_emb,
                                         top_k=top_k)
        self.encoder = get_encoder(encoder_type,
                                   d_emb=d_emb,
                                   n_blocks=n_blocks,
                                   n_heads=n_heads,
                                   bias_mha=bias_mha,
                                   dropout_mha=dropout_mha,
                                   bias_mlp=bias_mlp,
                                   dropout_mlp=dropout_mlp,
                                   h2i_ratio=h2i_ratio,
                                   with_edge=True)

        # Graph context
        self.graph_encoder = MLP(d_emb, h2i_ratio * d_emb, d_emb)
        # Layer index context
        self.layer_index_encoder = nn.Embedding(100, d_emb)
        # Layer variable context
        self.layer_var_encoder = nn.Embedding(100, d_emb)
        # State
        self.aggregator = MLP(d_emb, h2i_ratio * d_emb, d_emb)

        self.predictor = nn.Linear(d_emb, 2)

    def forward(self, n_feat, e_feat, pids_index, lids, vids, indices):
        # Tokenize
        n_feat, p_feat = self.tokenizer.tokenize(n_feat, e=e_feat)
        # Embed
        n_emb, e_emb = self.token_emb(n_feat, e_feat.int(), p_feat.float())
        # Encode: B' x n_vars x d_emb
        n_emb, _ = self.encoder(n_emb, e_emb)
        # pad 0 to n_emb so that -1 results in zero vec
        B_prime, _, d_emb = n_emb.shape
        # B' x (n_vars + 1) x d_emb
        n_emb = torch.cat((n_emb, torch.zeros((B_prime, 1, d_emb)).to(self.device)), dim=1)

        # Instance embedding
        # B' x d_emb
        inst_emb = self.graph_encoder(n_emb.sum(1))
        # B x d_emb
        inst_emb = torch.stack([inst_emb[pid] for pid in pids_index])

        # Layer-index embedding
        # B x d_emb
        li_emb = self.layer_index_encoder(lids)

        # Layer-variable embedding
        # B x d_emb
        lv_emb = torch.stack([n_emb[pid, vid] for pid, vid in zip(pids_index, vids.int())])

        # State embedding
        n_emb = torch.stack([n_emb[pid] for pid in pids_index])
        state_emb = torch.stack([n_emb[pid][state].sum(0) for pid, state in zip(pids_index, indices)])

        # for ibatch, states in enumerate(indices):
        #     state_emb.append(torch.stack([n_feat[ibatch][state].sum(0)
        #                                   for state in states]))
        # state_emb = torch.stack(state_emb)
        state_emb = self.aggregator(state_emb)
        context = inst_emb + li_emb + lv_emb
        state_emb = state_emb + context

        # Pareto-state predictor
        logits = self.predictor(state_emb)

        return logits


class ParetoStatePredictorKnapsack(nn.Module):
    def __init__(self,
                 encoder_type="transformer",
                 n_obj_feat=2,
                 n_con_feat=2,
                 d_emb=64,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 device=None):
        super(ParetoStatePredictorKnapsack, self).__init__()
        self.tokenizer = KnapsackInstanceTokenizer(device=device)
        self.token_emb = TokenEmbedKnapsack(n_obj_feat,
                                            n_con_feat,
                                            d_emb)
        self.encoder = get_encoder(encoder_type,
                                   d_emb=d_emb,
                                   n_blocks=n_blocks,
                                   n_heads=n_heads,
                                   bias_mha=bias_mha,
                                   dropout_mha=dropout_mha,
                                   bias_mlp=bias_mlp,
                                   dropout_mlp=dropout_mlp,
                                   h2i_ratio=h2i_ratio,
                                   with_edge=False)

        # Graph context
        self.instance_encoder = MLP(d_emb, h2i_ratio * d_emb, d_emb)
        # Layer index context
        self.layer_index_encoder = nn.Embedding(100, d_emb)
        # Layer variable context
        self.layer_var_encoder = nn.Embedding(100, d_emb)
        # State
        self.aggregator = MLP(1, h2i_ratio * d_emb, d_emb)

        self.predictor = nn.Linear(d_emb, 2)

    def forward(self, n_feat, lids, vids, states):
        # Tokenize
        o, c = self.tokenizer.tokenize(n_feat)
        # Embed
        n_emb = self.token_emb(o, c)
        # Encode
        n_emb, _ = self.encoder(n_emb)

        # Instance embedding
        inst_emb = self.instance_encoder(n_emb.sum(1))
        # Layer-index embedding
        li_emb = self.layer_index_encoder(lids)
        # Layer-variable embedding
        lv_emb = n_emb[torch.arange(n_feat.shape[0]), vids.int(), :]
        # State embedding
        state_emb = self.aggregator(states)
        state_emb = state_emb + (inst_emb + li_emb + lv_emb).unsqueeze(1)

        # Pareto-state predictor
        logits = self.predictor(state_emb)

        return logits
