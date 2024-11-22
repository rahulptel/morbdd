import torch
import torch.nn as nn


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


class GTFParetoStatePredictor(nn.Module):
    def __init__(
        self,
        n_node_feat=2,
        n_edge_type=2,
        d_emb=64,
        top_k=5,
        n_layers=2,
        n_heads=8,
        dropout_token=0.0,
        dropout_attn=0.1,
        dropout_proj=0.1,
        dropout_mlp=0.1,
        bias_mha=False,
        bias_mlp=False,
        h2i_ratio=2,
    ):
        super(GTFParetoStatePredictor, self).__init__()
        print("Model: Graph Transformer")
        self.token_emb = TokenEmbedGraph(
            n_node_feat,
            n_edge_type=n_edge_type,
            d_emb=d_emb,
            top_k=top_k,
            dropout=dropout_token,
        )
        self.node_encoder = GTEncoder(
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


class ParetoStatePredictorMIS(nn.Module):
    def __init__(
        self,
        encoder_type="transformer",
        n_node_feat=2,
        n_edge_type=2,
        d_emb=64,
        top_k=5,
        n_layers=2,
        n_heads=8,
        dropout_token=0.2,
        dropout=0.2,
        bias_mha=False,
        bias_mlp=False,
        h2i_ratio=2,
    ):
        super(ParetoStatePredictorMIS, self).__init__()
        self.encoder_type = encoder_type
        self.token_emb = TokenEmbedGraph(
            encoder_type,
            n_node_feat,
            n_edge_type=n_edge_type,
            d_emb=d_emb,
            top_k=top_k,
            dropout=dropout,
        )
        self.set_node_encoder(
            d_emb=d_emb,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout_token=dropout_token,
            bias_mha=bias_mha,
            dropout=dropout,
            bias_mlp=bias_mlp,
            h2i_ratio=h2i_ratio,
        )
        assert self.node_encoder is not None

        # Graph context
        self.graph_encoder = MLP(d_emb, d_emb, d_emb, dropout=dropout)
        # Layer index context
        self.layer_index_encoder = MLP(1, d_emb, d_emb, dropout=dropout)
        # self.layer_index_encoder = nn.Embedding(100, d_emb)
        # State
        self.aggregator = MLP(d_emb, h2i_ratio * d_emb, d_emb, dropout=dropout)

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

    def set_node_encoder(
        self,
        d_emb=64,
        n_layers=2,
        n_heads=8,
        dropout=0.2,
        dropout_token=0.0,
        bias_mha=False,
        bias_mlp=False,
        h2i_ratio=2,
    ):
        if self.encoder_type == "transformer":
            print("Using Graph Transformer")
            self.node_encoder = GTEncoder(
                d_emb=d_emb,
                n_layers=n_layers,
                n_heads=n_heads,
                bias_mha=bias_mha,
                dropout_mha=dropout,
                bias_mlp=bias_mlp,
                dropout_mlp=dropout,
                h2i_ratio=h2i_ratio,
            )
        elif self.encoder_type == "gat":
            print("Using GAT Encoder")
            self.node_encoder = GATEncoder(
                d_emb=d_emb, n_layers=n_layers, n_heads=n_heads, dropout=dropout
            )
        else:
            print("Invalid node encoder!")
            self.node_encoder = None
