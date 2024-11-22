import torch
import torch.nn as nn


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

    def __init__(self, cfg, is_last_block=False):
        super(MultiHeadSelfAttentionWithEdge, self).__init__()
        assert cfg.d_emb % cfg.n_heads == 0

        self.d_emb = cfg.d_emb
        self.d_k = cfg.d_emb // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.is_last_block = is_last_block
        self.drop_attn = nn.Dropout(cfg.dropout_attn)
        self.drop_proj_n = nn.Dropout(cfg.dropout_proj)

        # Node Q, K, V params
        self.W_qkv = nn.Linear(cfg.d_emb, 3 * cfg.d_emb, bias=cfg.bias_mha)
        self.O_n = nn.Linear(cfg.n_heads * self.d_k, cfg.d_emb, bias=cfg.bias_mha)

        # Edge bias and gating parameters
        self.W_g = nn.Linear(cfg.d_emb, cfg.n_heads, bias=cfg.bias_mha)
        self.W_e = nn.Linear(cfg.d_emb, cfg.n_heads, bias=cfg.bias_mha)

        # Output mapping params
        if is_last_block:
            self.O_e = None
        else:
            self.O_e = nn.Linear(cfg.n_heads, cfg.d_emb, bias=cfg.bias_mha)
            self.drop_proj_e = nn.Dropout(cfg.dropout_proj)

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
    def __init__(self, cfg, is_last_block=False):
        super(GTEncoderLayer, self).__init__()
        self.is_last_block = is_last_block
        # MHA with edge information
        self.ln_n1 = nn.LayerNorm(cfg.d_emb)
        self.ln_e1 = nn.LayerNorm(cfg.d_emb)
        self.mha = MultiHeadSelfAttentionWithEdge(cfg, is_last_block=self.is_last_block)
        # FF
        self.ln_n2 = nn.LayerNorm(cfg.d_emb)
        self.mlp_node = MLP(
            cfg.d_emb,
            cfg.h2i_ratio * cfg.d_emb,
            cfg.d_emb,
            bias=cfg.bias_mlp,
            normalize=False,
            dropout=0.0,
        )
        self.dropout_mlp_n = nn.Dropout(cfg.dropout_mlp)

        if not is_last_block:
            # self.dropout_mha_e = nn.Dropout(dropout_mha)
            self.ln_e2 = nn.LayerNorm(cfg.d_emb)
            self.mlp_edge = MLP(
                cfg.d_emb,
                cfg.h2i_ratio * cfg.d_emb,
                cfg.d_emb,
                bias=cfg.bias_mlp,
                normalize=False,
                dropout=0.0,
            )
            self.dropout_mlp_e = nn.Dropout(cfg.dropout_mlp)

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
    def __init__(self, cfg):
        super(GTEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            is_last_block = i == cfg.n_layers - 1
            self.encoder_blocks.append(GTEncoderLayer(cfg, is_last_block=is_last_block))

    def forward(self, n, e):
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n


class MultiHeadSelfAttention(nn.Module):
    """Based on: Attention is all you need"""

    def __init__(self, cfg):
        super(MultiHeadSelfAttention, self).__init__()
        assert cfg.d_emb // cfg.n_heads == 0
        self.d_k = cfg.d_emb // cfg.n_heads
        self.d_emb = cfg.d_emb
        self.n_heads = cfg.n_heads
        # Node Q, K, V params
        self.W_q = nn.Linear(cfg.d_emb, cfg.n_heads * self.d_k, bias=cfg.bias_mha)
        self.W_k = nn.Linear(cfg.d_emb, cfg.n_heads * self.d_k, bias=cfg.bias_mha)
        self.W_v = nn.Linear(cfg.d_emb, cfg.n_heads * self.d_k, bias=cfg.bias_mha)
        self.O_n = nn.Linear(cfg.n_heads * self.d_k, cfg.d_emb, bias=cfg.bias_mha)

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
        _A = torch.einsum("ijkl,ijlm->ijkm", [Q, K.transpose(-2, -1)])
        _A = _A * ((self.d_k) ** (-0.5))
        _A = torch.clamp(_A, -5, 5)
        _A = torch.softmax(_A, dim=-1)

        # batch_size x n_heads x n_nodes x d_k
        _V = _A @ V
        # batch_size x n_nodes x d_emb
        n = self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb))

        return n


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        # MHA
        self.ln_n1 = nn.LayerNorm(cfg.d_emb)
        self.mha = MultiHeadSelfAttention(cfg)
        self.dropout_mha = nn.Dropout(cfg.dropout_mha)
        # FF
        self.ln_n2 = nn.LayerNorm(cfg.d_emb)
        self.mlp_node = MLP(
            cfg.d_emb,
            cfg.h2i_ratio * cfg.d_emb,
            cfg.d_emb,
            bias=cfg.bias_mlp,
            normalize=False,
            dropout=0.0,
            act=cfg.act,
        )
        self.dropout_mlp = nn.Dropout(cfg.dropout_mlp)

    def forward(self, n):
        n = self.ln_n1(n)
        n = n + self.dropout_mha(self.mha(n))

        n = self.ln_n2(n)
        n = n + self.dropout_mlp(self.mlp_node(n))

        return n


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.n_layers)]
        )

    def forward(self, n):
        for block in self.encoder_blocks:
            n = block(n)

        return n
