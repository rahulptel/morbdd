import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MLP


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
        o = torch.cat(
            (objs.transpose(1, 2).unsqueeze(-1), obj_id.unsqueeze(-1)), dim=-1
        )

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


class TFParetoStatePredictor(nn.Module):
    def __init__(
        self,
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
        device=None,
    ):
        super(TFParetoStatePredictor, self).__init__()
        print("Model: Transformer")

        self.tokenizer = KnapsackInstanceTokenizer(device=device)
        self.token_emb = TokenEmbedKnapsack(n_obj_feat, n_con_feat, d_emb)
        self.encoder = self.get_encoder(
            encoder_type,
            d_emb=d_emb,
            n_layers=n_layers,
            n_heads=n_heads,
            bias_mha=bias_mha,
            dropout_mha=dropout_mha,
            bias_mlp=bias_mlp,
            dropout_mlp=dropout_mlp,
            h2i_ratio=h2i_ratio,
            with_edge=False,
        )

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


class ParetoStatePredictorKnapsack(nn.Module):
    def __init__(
        self,
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
        device=None,
    ):
        super(ParetoStatePredictorKnapsack, self).__init__()
        self.tokenizer = KnapsackInstanceTokenizer(device=device)
        self.token_emb = TokenEmbedKnapsack(n_obj_feat, n_con_feat, d_emb)
        self.encoder = self.get_encoder(
            encoder_type,
            d_emb=d_emb,
            n_layers=n_layers,
            n_heads=n_heads,
            bias_mha=bias_mha,
            dropout_mha=dropout_mha,
            bias_mlp=bias_mlp,
            dropout_mlp=dropout_mlp,
            h2i_ratio=h2i_ratio,
            with_edge=False,
        )

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
