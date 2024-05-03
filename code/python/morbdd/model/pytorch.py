import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MISInputEncoder(nn.Module):
    """
    Tokenize graph input to obtain position-aware node embedding and
    edge embeddings
    """

    def __init__(self, d_emb=64, top_k=5):
        super(MISInputEncoder, self).__init__()

        self.top_k = top_k
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, d_emb)
        self.edge_encoder = nn.Embedding(2, d_emb)
        self.pos_encoder = nn.Linear(top_k * 2, d_emb)

    def forward(self, n, e):
        # Calculate position encoding
        U, S, Vh = torch.linalg.svd(e.float())
        U = U[:, :, :self.top_k]
        S = (torch.diag_embed(S)[:, :self.top_k, :self.top_k]) ** (1 / 2)
        Vh = Vh[:, :self.top_k, :]

        L, R = U @ S, S @ Vh
        R = R.permute(0, 2, 1)
        p = torch.cat((L, R), dim=-1)  # B x n_vars x (2*top_k)
        p = self.pos_encoder(p)  # B x n_vars x d_emb

        # Calculate node and edge encodings
        B, n_objs, n_vars = n.shape

        obj_id = torch.arange(1, n_objs + 1) / 10
        obj_id = obj_id.repeat((n_vars, 1))
        obj_id = obj_id.repeat((B, 1, 1)).to(torch.device('cuda'))
        n = torch.cat((n.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1)
        n = n.transpose(1, 2)
        n = F.relu(self.linear1(n))  # B x n_objs x n_vars x d_emb
        # Sum aggregate objectives
        n = n.sum(1)  # B x n_vars x d_emb
        n = F.relu(self.linear2(n))
        # Update node encoding with positional encoding based on SVD
        n = n + p
        e = self.edge_encoder(e)  # B x n_vars x n_vars x d_emb

        return n, e


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


class MultiHeadSelfAttention(nn.Module):
    """
    Based on: Global Self-Attention as a Replacement for Graph Convolution
    https://arxiv.org/pdf/2108.03348
    """

    def __init__(self,
                 d_emb=64,
                 n_heads=8,
                 bias_mha=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_k = d_emb // n_heads
        self.d_emb = d_emb
        self.n_heads = n_heads
        # Node Q, K, V params
        self.W_q = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_k = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        self.W_v = nn.Linear(d_emb, n_heads * self.d_k, bias=bias_mha)
        # Edge bias and gating parameters
        self.W_g = nn.Linear(d_emb, n_heads, bias=bias_mha)
        self.W_e = nn.Linear(d_emb, n_heads, bias=bias_mha)
        # Output mapping params
        self.O_e = nn.Linear(n_heads, d_emb, bias=bias_mha)
        self.O_n = nn.Linear(n_heads * self.d_k, d_emb, bias=bias_mha)

    def forward(self, n, e):
        """
        n : batch_size x n_nodes x d_emb
        e : batch_size x n_nodes x n_nodes x d_emb
        """
        batch_size = n.shape[0]

        # Compute QKV and reshape
        # batch_size x n_nodes x (n_heads * d_k)
        Q, K, V = self.W_q(n), self.W_k(n), self.W_v(n)
        # batch_size x n_nodes x n_heads x d_k
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)
        # batch_size x n_heads x n_nodes x d_k
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

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

        n = self.O_n(_V.transpose(1, 2).reshape(batch_size, -1, self.d_emb))
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
                 node_residual=True,
                 edge_residual=True):
        super(GTEncoderLayer, self).__init__()

        self.ln_n1 = nn.LayerNorm(d_emb)
        self.ln_e1 = nn.LayerNorm(d_emb)
        self.mha = MultiHeadSelfAttention(d_emb=d_emb,
                                          n_heads=n_heads,
                                          bias_mha=bias_mha)
        self.dropout_mha = nn.Dropout(dropout_mha)

        self.ln_n2 = nn.LayerNorm(d_emb)
        self.ln_e2 = nn.LayerNorm(d_emb)
        self.mlp_node = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp)
        self.mlp_edge = MLP(d_emb, h2i_ratio * d_emb, d_emb, bias=bias_mlp)
        self.dropout_mlp = nn.Dropout(dropout_mlp)

    def forward(self, n, e):
        n, e = self.ln_n1(n), self.ln_e1(e)
        n_, e_ = self.mha(n, e)
        n, e = n + self.dropout_mha(n_), e + self.dropout_mha(e_)

        n, e = self.ln_n2(n), self.ln_e2(e)
        n, e = n + self.dropout_mlp(self.mlp_node(n)), e + self.dropout_mlp(self.mlp_edge(e))

        return n, e


class GTEncoder(nn.Module):
    def __init__(self,
                 d_emb=64,
                 top_k=5,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 node_residual=True,
                 edge_residual=True):
        super(GTEncoder, self).__init__()
        self.input_encoder = MISInputEncoder(d_emb=d_emb, top_k=top_k)
        self.encoder_blocks = clones(GTEncoderLayer(d_emb=d_emb,
                                                    n_heads=n_heads,
                                                    bias_mha=bias_mha,
                                                    dropout_mha=dropout_mha,
                                                    bias_mlp=bias_mlp,
                                                    dropout_mlp=dropout_mlp,
                                                    h2i_ratio=h2i_ratio,
                                                    node_residual=node_residual,
                                                    edge_residual=edge_residual),
                                     n_blocks)

    def forward(self, n_feat, e_feat):
        n, e = self.input_encoder(n_feat, e_feat)
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n, e


class ParetoStatePredictor(nn.Module):
    def __init__(self,
                 d_emb=64,
                 top_k=5,
                 n_blocks=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 node_residual=True,
                 edge_residual=True):
        super(ParetoStatePredictor, self).__init__()
        self.encoder = GTEncoder(d_emb=d_emb,
                                 top_k=top_k,
                                 n_blocks=n_blocks,
                                 n_heads=n_heads,
                                 bias_mha=bias_mha,
                                 dropout_mha=dropout_mha,
                                 bias_mlp=bias_mlp,
                                 dropout_mlp=dropout_mlp,
                                 h2i_ratio=h2i_ratio,
                                 node_residual=node_residual,
                                 edge_residual=edge_residual)
        # Graph context
        self.norm = nn.LayerNorm(d_emb)
        self.graph_encoder = MLP(d_emb, h2i_ratio * d_emb, d_emb)
        # Layer index context
        self.layer_index_encoder = nn.Embedding(100, d_emb)
        # Layer variable context
        self.layer_var_encoder = nn.Embedding(100, d_emb)
        # State
        self.aggregator = MLP(d_emb, h2i_ratio * d_emb, d_emb)

        self.predictor = nn.Linear(d_emb, 2)

    def forward(self, n_feat, e_feat, lids, vids, indices):
        n_feat, e_feat = self.encoder(n_feat, e_feat)
        # pad 0 to n_feat so that
        B, _, n_emb = n_feat.shape
        n_feat = torch.cat((n_feat, torch.zeros((B, 1, n_emb)).to(torch.device('cuda'))), dim=1)

        g_emb = self.graph_encoder(n_feat.sum(1))
        li_emb = self.layer_index_encoder(lids)
        lv_emb = n_feat[torch.arange(n_feat.shape[0]), vids.int(), :]
        context = (g_emb + li_emb + lv_emb).unsqueeze(1)

        state_emb = []
        for ibatch, states in enumerate(indices):
            state_emb.append(torch.stack([n_feat[ibatch][state].sum(0)
                                          for state in states]))
        state_emb = torch.stack(state_emb)
        state_emb = self.aggregator(state_emb)
        state_emb = state_emb + context

        logits = self.predictor(state_emb)
        return logits

#
#
# class ParetoNodePredictor(nn.Module):
#     def __init__(self, cfg):
#         super(ParetoNodePredictor, self).__init__()
#         self.cfg = cfg
#         self.variable_encoder = GraphVariableEncoder(cfg)
#         # self.state_encoder = StateEncoder(cfg)
#         self.layer_encoding = nn.Parameter(torch.randn(cfg.prob.n_vars, cfg.n_emb))
#
#         self.ff = nn.ModuleList()
#         self.ff.append(nn.LayerNorm(cfg.n_emb))
#         self.ff.append(nn.Dropout(cfg.dropout))
#         if cfg.agg == "sum":
#             self.ff.append(nn.Linear(cfg.n_emb, 2))
#         if cfg.agg == "cat":
#             self.ff.append(nn.Linear(4 * cfg.n_emb, 2))
#         self.ff = nn.Sequential(*self.ff)
#
#     def forward(self, node_feat, var_feat, adj, var_id):
#         # B x n_var_mis x n_emb
#         var_enc = self.variable_encoder(var_feat, adj)
#
#         # B x n_emb
#         graph_enc = var_enc.sum(dim=1)
#
#         # B x n_emb
#         B, nV, nF = var_enc.shape
#         layer_var_enc = var_enc[torch.arange(B), var_id.int(), :].squeeze(1)
#
#         # B x n_emb
#         layer_enc = self.layer_encoding[node_feat[:, 0].view(-1).int()]
#
#         # B x n_emb
#         active_vars = node_feat[:, 1:].bool()
#         state_enc = torch.stack([ve[active_vars[i]].sum(dim=0)
#                                  for i, ve in enumerate(var_enc)])
#
#         x = None
#         if self.cfg.agg == "sum":
#             x = graph_enc + layer_var_enc + layer_enc + state_enc
#         elif self.cfg.agg == "cat":
#             x = torch.cat((graph_enc, layer_var_enc, layer_enc, state_enc), dim=-1)
#
#         x = self.ff(x)
#         return x

# class FeedForwardUnit(nn.Module):
#     def __init__(self, ni, nh, no, dropout=0.0, bias=True, activation="relu"):
#         super(FeedForwardUnit, self).__init__()
#         self.i2h = nn.Linear(ni, nh, bias=bias)
#         self.h2o = nn.Linear(nh, no, bias=bias)
#         if activation == "relu":
#             self.act = nn.ReLU()
#         elif activation == "gelu":
#             self.act = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x = self.i2h(x)
#         x = self.act(x)
#         x = self.h2o(x)
#         x = self.dropout(x)
#
#         return x
#
#
# class GraphForwardUnit(nn.Module):
#     def __init__(self, ni, nh, no, dropout=0.0, bias=True, activation="relu"):
#         super(GraphForwardUnit, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.ln_1 = nn.LayerNorm(ni)
#         self.ffu = FeedForwardUnit(ni, nh, no, dropout=dropout, bias=bias, activation=activation)
#
#         self.i2o = None
#         if ni != no:
#             self.i2o = nn.Linear(ni, no, bias=False)
#         if activation == "relu":
#             self.act = nn.ReLU()
#         elif activation == "gelu":
#             self.act = nn.GELU()
#
#         self.ln_2 = nn.LayerNorm(no)
#         self.pos = nn.Linear(no, no, bias)
#
#         self.ln_3 = nn.LayerNorm(no)
#         self.neg = nn.Linear(no, no, bias)
#         self.combine = nn.Linear(2 * no, no, bias)
#
#     def forward(self, x, adj):
#         neg_adj = 1 - adj
#
#         ffu_x = self.ffu(self.ln_1(x))
#         x = x if self.i2o is None else self.i2o(x)
#         x = x + ffu_x
#
#         x_pos = self.act(self.pos(self.ln_2(adj @ x)))
#         x_neg = self.act(self.neg(self.ln_3(neg_adj @ x)))
#         x = x + self.act(self.combine(self.dropout(torch.cat((x_pos, x_neg), dim=-1))))
#
#         return x
#
#
# class GraphVariableEncoder(nn.Module):
#     def __init__(self, cfg):
#         super(GraphVariableEncoder, self).__init__()
#         self.cfg = cfg
#         self.n_layers = self.cfg.graph_enc.n_layers
#         n_feat = self.cfg.graph_enc.n_feat
#         n_emb = self.cfg.graph_enc.n_emb
#         act = self.cfg.graph_enc.activation
#         dp = self.cfg.graph_enc.dropout
#         bias = self.cfg.graph_enc.bias
#
#         self.units = nn.ModuleList()
#         for layer in range(self.n_layers):
#             if layer == 0:
#                 self.units.append(GraphForwardUnit(n_feat, n_emb, n_emb, dropout=dp, activation=act, bias=bias))
#             else:
#                 self.units.append(GraphForwardUnit(n_emb, n_emb, n_emb, dropout=dp, activation=act, bias=bias))
#
#     def forward(self, x, adj):
#         for layer in range(self.n_layers):
#             x = self.units[layer](x, adj)
#
#         return x

# class SetEncoder(nn.Module):
#     def __init__(self, enc_net_dims=[2, 4], agg_net_dims=[4, 8]):
#         super(SetEncoder, self).__init__()
#         self.enc_net_dims = enc_net_dims
#         self.agg_net_dims = agg_net_dims
#
#         self.encoder_net = nn.ModuleList()
#         for i in range(1, len(self.enc_net_dims)):
#             self.encoder_net.append(nn.Linear(self.enc_net_dims[i - 1],
#                                               self.enc_net_dims[i]))
#             self.encoder_net.append(nn.ReLU())
#         self.encoder_net = nn.Sequential(*self.encoder_net)
#
#         self.aggregator_net = nn.ModuleList()
#         for i in range(1, len(self.agg_net_dims)):
#             self.aggregator_net.append(nn.Linear(self.agg_net_dims[i - 1],
#                                                  self.agg_net_dims[i]))
#             self.aggregator_net.append(nn.ReLU())
#         self.aggregator_net = nn.Sequential(*self.aggregator_net)
#
#     def forward(self, x):
#         # print(x.shape)
#         x_enc = self.encoder_net(x)
#         # print(x1.shape)
#         x_enc_agg = torch.sum(x_enc, axis=1)
#         # print(x1_agg.shape)
#
#         return self.aggregator_net(x_enc_agg)
#
#
# class ParetoStatePredictor(nn.Module):
#     def __init__(self, cfg):
#         super(ParetoStatePredictor, self).__init__()
#         self.cfg = cfg
#         self.instance_encoder = SetEncoder(list(self.cfg.ie.enc), list(self.cfg.ie.agg))
#         self.context_encoder = SetEncoder(list(self.cfg.ce.enc), list(self.cfg.ce.agg))
#         self.parent_encoder = SetEncoder(list(self.cfg.pe.enc), list(self.cfg.pe.agg))
#
#         self.node_encoder = nn.ModuleList()
#         for i in range(1, len(self.cfg.ne)):
#             self.node_encoder.append(nn.Linear(self.cfg.ne[i - 1], self.cfg.ne[i]))
#             self.node_encoder.append(nn.ReLU())
#         self.node_encoder = nn.Sequential(*self.node_encoder)
#
#         pred_in_dim = self.cfg.ie.agg[-1] + self.cfg.ce.agg[-1] + self.cfg.pe.agg[-1] + self.cfg.ne[-1]
#         self.predictor = nn.Sequential(nn.Linear(pred_in_dim, 1),
#                                        nn.Sigmoid())
#
#     def forward(self, instf, vf, nf, pf):
#         # print(instf.shape, vf.shape, nf.shape, pf.shape)
#         # print(self.node_encoder)
#         ie = self.instance_encoder(instf)
#         # print(ie.shape)
#         ve = self.context_encoder(vf)
#         # print(ve.shape)
#         pe = self.parent_encoder(pf)
#         # print(pe.shape)
#         ne = self.node_encoder(nf)
#         # print(ne.shape)
#         emb = torch.concat((ie, ve, ne, pe), dim=1)
#         # print(emb.shape)
#
#         return self.predictor(emb)
