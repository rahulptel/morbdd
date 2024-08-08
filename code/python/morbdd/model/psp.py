import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .gtf import TokenEmbedGraph, GTEncoder


class TFParetoStatePredictor(nn.Module):
    def __init__(self,
                 encoder_type="transformer",
                 n_obj_feat=2,
                 n_con_feat=2,
                 d_emb=64,
                 n_layers=2,
                 n_heads=8,
                 bias_mha=False,
                 dropout_mha=0,
                 bias_mlp=True,
                 dropout_mlp=0.1,
                 h2i_ratio=2,
                 device=None):
        super(TFParetoStatePredictor, self).__init__()
        self.tokenizer = KnapsackInstanceTokenizer(device=device)
        self.token_emb = TokenEmbedKnapsack(n_obj_feat,
                                            n_con_feat,
                                            d_emb)
        self.encoder = self.get_encoder(encoder_type,
                                        d_emb=d_emb,
                                        n_layers=n_layers,
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


class GTFParetoStatePredictor(nn.Module):
    def __init__(self,
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
                 h2i_ratio=2):
        super(GTFParetoStatePredictor, self).__init__()
        print("Using Graph Transformer")
        self.token_emb = TokenEmbedGraph(n_node_feat, n_edge_type=n_edge_type, d_emb=d_emb, top_k=top_k,
                                         dropout=dropout_token)
        self.node_encoder = GTEncoder(d_emb=d_emb,
                                      n_layers=n_layers,
                                      n_heads=n_heads,
                                      bias_mha=bias_mha,
                                      dropout_attn=dropout_attn,
                                      dropout_proj=dropout_proj,
                                      bias_mlp=bias_mlp,
                                      dropout_mlp=dropout_mlp,
                                      h2i_ratio=h2i_ratio)

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
