import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class TokenEmbedGraph(nn.Module):
    """
    Tokenize graph input to obtain position-aware node embedding and
    edge embeddings
    """

    def __init__(self, n_node_feat, n_edge_type=2, d_emb=64, top_k=5, dropout=0.0):
        super(TokenEmbedGraph, self).__init__()
        self.n_edge_type = n_edge_type
        self.top_k = top_k
        self.linear1 = nn.Linear(n_node_feat, 2 * d_emb)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(2 * d_emb, d_emb)
        self.dropout2 = nn.Dropout(dropout)

        self.pos_encoder = nn.Linear(top_k * 2, d_emb)
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


class GTEncoder(nn.Module):
    def __init__(self,
                 d_emb=64,
                 n_layers=2,
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
                                                            is_last_block=i == n_layers - 1)
                                             for i in range(n_layers)])

    def forward(self, n, e):
        # print("EncoderBlock: ", n[0][0])
        for block in self.encoder_blocks:
            n, e = block(n, e)
            # print("EncoderBlock: ", n[0][0])

        return n
