import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.nn import GATConv


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


class TokenEmbedGraph(nn.Module):
    """
    Tokenize graph input to obtain position-aware node embedding and
    edge embeddings
    """

    def __init__(self, encoder_type, n_node_feat, n_edge_type=2, d_emb=64, top_k=0, dropout=0.0):
        super(TokenEmbedGraph, self).__init__()
        self.encoder_type = encoder_type
        self.n_edge_type = n_edge_type
        self.top_k = top_k
        self.linear1 = nn.Linear(n_node_feat, 2 * d_emb)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(2 * d_emb, d_emb)
        self.dropout2 = nn.Dropout(dropout)

        if self.top_k:
            self.pos_encoder = nn.Linear(top_k * 2, d_emb)

        if self.encoder_type == "transformer":
            self.edge_encoder = nn.Embedding(n_edge_type, d_emb)
            self.dropout3 = nn.Dropout(dropout)

    def forward(self, n, e, p):
        # Calculate node and edge encodings
        n = self.dropout1(F.relu(self.linear1(n)))  # B x n_vars x n_objs x 2 * d_emb
        # Sum aggregate objectives
        n = n.sum(2)  # B x n_vars x 2 * d_emb
        n_enc = self.dropout2(F.relu(self.linear2(n)))  # B x n_vars x d_emb

        # Update node encoding with positional encoding based on SVD
        if self.top_k:
            p = self.pos_encoder(p)  # B x n_vars x d_emb
            n_enc = n_enc + p

        e_enc = e
        if self.encoder_type == "transformer":
            e_enc = self.dropout3(self.edge_encoder(e))  # B x n_vars x n_vars x d_emb

        return n_enc, e_enc


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
        # self.W_q = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        # self.W_k = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        # self.W_v = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
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

        n = self.drop_proj_n(self.O_n(_V.transpose(1, 2))).reshape(B, -1, self.d_emb)
        e = None if self.O_e is None else self.drop_proj_e(self.O_e(_E.permute(0, 2, 3, 1)))

        return n, e


class GTEncoderLayer(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False,
                 dropout_attn=0.1,
                 dropout_proj=0.1,
                 bias_mlp=False,
                 dropout_mlp=0.2,
                 h2i_ratio=2,
                 is_last_block=False):
        super(GTEncoderLayer, self).__init__()
        self.is_last_block = is_last_block
        # MHA
        self.ln_n1 = nn.LayerNorm(d_emb)
        self.ln_e1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttentionWithEdge(d_emb=d_emb,
                                                  n_heads=n_heads,
                                                  bias_mha=bias_mha,
                                                  is_last_block=is_last_block,
                                                  dropout_attn=dropout_attn,
                                                  dropout_proj=dropout_proj)
        # self.dropout_mha_n = nn.Dropout(dropout_mha)
        # FF
        self.ln_n2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0)
        self.dropout_mlp_n = nn.Dropout(dropout_mlp)

        if not is_last_block:
            # self.dropout_mha_e = nn.Dropout(dropout_mha)
            self.ln_e2 = nn.LayerNorm(d_emb)
            self.mlp_edge = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0)
            self.dropout_mlp_e = nn.Dropout(dropout_mlp)

    def forward(self, n, e=None):
        n_, e_ = self.mha(self.ln_n1(n), self.ln_e1(e))
        # n = n + self.dropout_mha_n(n_)
        n += n_

        n = n + self.dropout_mlp_n(self.mlp_node(self.ln_n2(n)))
        if not self.is_last_block:
            # e = e + self.dropout_mha_e(e_)
            e += e_
            e = e + self.dropout_mlp_e(self.mlp_edge(self.ln_e2(e)))

        return n, e


class GATEncoder(nn.Module):
    def __init__(self, d_emb=64, n_blocks=2, n_heads=8, dropout=0.1):
        super(GATEncoder, self).__init__()
        d_k = d_emb // n_heads
        self.conv_list = nn.ModuleList([GATConv(d_emb, d_k, heads=n_heads, dropout=dropout)
                                        for _ in range(n_blocks)])

    def forward(self, n, adj_mat):
        n_nodes = n.shape[0]
        edge_index = [pyg.utils.dense_to_sparse(adj_mat[i])[0] for i in range(n_nodes)]
        for conv in self.conv_list:
            n = [F.relu(conv(n[i], edge_index[i])) for i in range(n_nodes)]

        n = torch.stack(n)
        return n


class GTEncoder(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_attn=0.1,
                 dropout_proj=0.1,
                 bias_mlp=False,
                 dropout_mlp=0.1,
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
                                                            is_last_block=i == n_blocks - 1)
                                             for i in range(n_blocks)])

    def forward(self, n, e):
        # print("EncoderBlock: ", n[0][0])
        for block in self.encoder_blocks:
            n, e = block(n, e)
            # print("EncoderBlock: ", n[0][0])

        return n


class ParetoStatePredictorMIS(nn.Module):
    def __init__(self,
                 encoder_type="transformer",
                 n_node_feat=2,
                 n_edge_type=2,
                 d_emb=64,
                 top_k=5,
                 n_blocks=2,
                 n_heads=8,
                 dropout_token=0.0,
                 dropout_attn=0.1,
                 dropout_proj=0.1,
                 dropout_mlp=0.1,
                 bias_mha=False,
                 bias_mlp=False,
                 h2i_ratio=2):
        super(ParetoStatePredictorMIS, self).__init__()
        self.encoder_type = encoder_type
        self.token_emb = TokenEmbedGraph(encoder_type, n_node_feat, n_edge_type=n_edge_type, d_emb=d_emb, top_k=top_k,
                                         dropout=dropout_token)
        self.set_node_encoder(d_emb=d_emb, n_blocks=n_blocks, n_heads=n_heads, dropout_attn=dropout_attn,
                              dropout_proj=dropout_proj, bias_mha=bias_mha, dropout_mlp=dropout_mlp, bias_mlp=bias_mlp,
                              h2i_ratio=h2i_ratio)
        assert self.node_encoder is not None

        # Graph context
        self.graph_encoder = MLP(d_emb, d_emb, d_emb, dropout=dropout_mlp)
        # Layer index context
        self.layer_index_encoder = MLP(1, d_emb, d_emb, dropout=dropout_mlp)
        # self.layer_index_encoder = nn.Embedding(100, d_emb)
        # State
        self.aggregator = MLP(d_emb, h2i_ratio * d_emb, d_emb, dropout=dropout_mlp)

        self.ln = nn.LayerNorm(d_emb)
        self.predictor = nn.Linear(d_emb, 2)

    def forward(self, n_feat, e_feat, pos_feat, lids, vids, states):
        # Embed
        n_emb, e_emb = self.token_emb(n_feat, e_feat.int(), pos_feat.float())
        # Encode: B x n_vars x d_emb
        n_emb = self.node_encoder(n_emb, e_emb)
        # Instance embedding
        # B x d_emb
        inst_emb = self.graph_encoder(n_emb.sum(1))
        # Layer-index embedding
        # B x d_emb
        li_emb = self.layer_index_encoder(lids.reshape(-1, 1).float())
        # Layer-variable embedding
        # B x d_emb
        lv_emb = n_emb[torch.arange(vids.shape[0]), vids.int()]
        # State embedding
        state_emb = torch.einsum("ijk,ij->ik", [n_emb, states.float()])
        state_emb = self.aggregator(state_emb)
        state_emb = state_emb + inst_emb + li_emb + lv_emb
        # Pareto-state predictor
        logits = self.predictor(self.ln(state_emb))

        return logits

    def set_node_encoder(self, d_emb=64, n_blocks=2, n_heads=8, dropout_mlp=0.1, dropout_attn=0.1, dropout_proj=0.1,
                         bias_mha=False, bias_mlp=False, h2i_ratio=2):
        if self.encoder_type == "transformer":
            print("Using Graph Transformer")
            self.node_encoder = GTEncoder(d_emb=d_emb,
                                          n_blocks=n_blocks,
                                          n_heads=n_heads,
                                          bias_mha=bias_mha,
                                          dropout_attn=dropout_attn,
                                          dropout_proj=dropout_proj,
                                          bias_mlp=bias_mlp,
                                          dropout_mlp=dropout_mlp,
                                          h2i_ratio=h2i_ratio)
        elif self.encoder_type == "gat":
            print("Using GAT Encoder")
            self.node_encoder = GATEncoder(d_emb=d_emb,
                                           n_blocks=n_blocks,
                                           n_heads=n_heads,
                                           dropout=dropout_mlp)
        else:
            print("Invalid node encoder!")
            self.node_encoder = None


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


class MultiHeadSelfAttention(nn.Module):
    """  Based on: Attention is all you need """

    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_emb // n_heads == 0
        self.d_k = d_emb // n_heads
        self.d_emb = d_emb
        self.n_heads = n_heads
        # Node Q, K, V params
        self.W_q = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_k = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_v = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.O_n = nn.Linear(n_heads * self.d_k, d_emb, bias=bias_mha)

    def forward(self, n):
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

        return n


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0.2,
                 bias_mlp=False,
                 dropout_mlp=0.2,
                 h2i_ratio=2):
        super(EncoderLayer, self).__init__()
        # MHA
        self.ln_n1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttention(d_emb=d_emb,
                                          n_heads=n_heads,
                                          bias_mha=bias_mha)
        self.dropout_mha = nn.Dropout(dropout_mha)
        # FF
        self.ln_n2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp, normalize=False, dropout=0.0)
        self.dropout_mlp = nn.Dropout(dropout_mlp)

    def forward(self, n):
        n = n + self.dropout_mha(self.mha(self.ln_n1(n)))
        n = n + self.dropout_mlp(self.mlp_node(self.ln_n2(n)))

        return n


class Encoder(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0.2,
                 bias_mlp=False,
                 dropout_mlp=0.2,
                 h2i_ratio=2):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([EncoderLayer(d_emb=d_emb,
                                                          n_heads=n_heads,
                                                          bias_mha=bias_mha,
                                                          dropout_mha=dropout_mha,
                                                          bias_mlp=bias_mlp,
                                                          dropout_mlp=dropout_mlp,
                                                          h2i_ratio=h2i_ratio)
                                             for _ in range(n_blocks)])

    def forward(self, n):
        for block in self.encoder_blocks:
            n = block(n)

        return n


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
        self.encoder = self.get_encoder(encoder_type,
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
        self.layer_index_encoder = MLP(1, d_emb, d_emb)
        # self.layer_index_encoder = nn.Embedding(100, d_emb)
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
