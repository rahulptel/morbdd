import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


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
                 n_layers=2,
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
                                             for _ in range(n_layers)])

    def forward(self, n):
        for block in self.encoder_blocks:
            n = block(n)

        return n
