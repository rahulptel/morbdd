import torch
import torch.nn as nn

from .base import GTEncoder


class GraphTokenizer(nn.Module):
    """
    DeepSet-based node and edge embeddings
    """

    def __init__(self, cfg):
        super(GraphTokenizer, self).__init__()
        self.linear1 = nn.Linear(cfg.n_node_feat, cfg.h2i_ratio * cfg.d_emb)
        self.linear2 = nn.Linear(cfg.h2i_ratio * cfg.d_emb, cfg.d_emb)
        self.linear3 = nn.Linear(cfg.n_edge_feat, cfg.d_emb)
        self.linear4 = nn.Linear(cfg.d_emb, cfg.d_emb)
        self.act = nn.ReLU() if cfg.act == "relu" else nn.GELU()

    def forward(self, n, e):
        n = self.act(self.linear1(n))  # B x n_objs x n_vars x (2 * d_emb)
        n = n.sum(1)  # B x n_vars x (2 * d_emb)
        n = self.act(self.linear2(n))  # B x n_vars x d_emb

        e = e.unsqueeze(-1)
        e = self.act(self.linear3(e))  # B x n_objs x n_vars x n_vars x d_emb
        e = e.sum(1)  # B x n_vars x n_vars x d_emb
        e = self.act(self.linear4(e))  # B x n_vars x n_vars x d_emb

        return n, e


class ParetoNodePredictor(nn.Module):
    # NOT_VISITED = 0
    # VISITED = 1
    # LAST_VISITED = 2
    NODE_VISIT_TYPES = 3
    N_LAYER_INDEX = 1
    N_CLASSES = 2

    def __init__(self, cfg):
        super(ParetoNodePredictor, self).__init__()
        self.concat_emb = cfg.concat_emb
        self.act = nn.ReLU() if cfg.act == "relu" else nn.GELU()
        self.graph_tokenizer = GraphTokenizer(cfg)
        self.graph_encoder = GTEncoder(cfg)
        self.visit_encoder = nn.Embedding(self.NODE_VISIT_TYPES, cfg.d_emb)
        self.node_visit_encoder1 = nn.Sequential(
            nn.Linear(cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
            self.act,
        )
        self.node_visit_encoder2 = nn.Sequential(
            nn.Linear(cfg.h2i_ratio * cfg.d_emb, cfg.d_emb),
            self.act,
        )
        self.layer_encoder = nn.Sequential(
            nn.Linear(self.N_LAYER_INDEX, cfg.d_emb),
            self.act,
        )
        if self.concat_emb:
            self.pareto_predictor = nn.Sequential(
                nn.Linear(3 * cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
                self.act,
                nn.Linear(cfg.h2i_ratio * cfg.d_emb, self.N_CLASSES),
            )
        else:
            self.pareto_predictor = nn.Sequential(
                nn.Linear(cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
                self.act,
                nn.Linear(cfg.h2i_ratio * cfg.d_emb, self.N_CLASSES),
            )

    def forward(self, n, e, l, s):
        n, e = self.graph_tokenizer(n, e)
        n = self.graph_encoder(n, e)  # B x n_vars x d_emb
        B, n_vars, d_emb = n.shape

        last_visit = s[:, -1]
        visit_mask = s[:, :-1]
        visit_mask[torch.arange(B), last_visit.long()] = 2
        visit_enc = self.visit_encoder(visit_mask.long())

        # B x d_emb
        node_visit = self.node_visit_encoder2(
            self.node_visit_encoder1((n + visit_enc)).sum(1)
        )
        customer_enc = n[torch.arange(B), last_visit.long()]
        l_enc = self.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))

        if self.concat_emb:
            return self.pareto_predictor(
                torch.cat((node_visit, customer_enc, l_enc), dim=-1)
            )
        else:
            return self.pareto_predictor(node_visit + customer_enc + l_enc)

    def configure_optimizer(self, cfg):
        params = self.parameters()
        if cfg.wd > 0:
            # Ref: https://github.com/karpathy/nanoGPT/blob/master/model.py
            # start with all the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": cfg.wd},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            self.print_params_info(
                decay_params, num_decay_params, nodecay_params, num_nodecay_params
            )
            params = optim_groups

        optimizer_cls = getattr(torch.optim, cfg.type)
        optimizer = optimizer_cls(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        print(f"using optimizer: {cfg.type}")
        print()

        return optimizer

    @staticmethod
    def print_params_info(
        decay_params, num_decay_params, nodecay_params, num_nodecay_params
    ):
        print(
            "num decayed parameter tensors: {}, with {} parameters".format(
                len(decay_params), num_decay_params
            )
        )
        print(
            "num non-decayed parameter tensors: {}, with {} parameters".format(
                len(nodecay_params), num_nodecay_params
            )
        )
