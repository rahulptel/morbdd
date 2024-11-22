import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import rankdata

from morbdd import ResourcePaths
from morbdd.scorer.scorer import NodeScorer

resource = ResourcePaths()


class TSPHeuristicNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.inst = None
        self.edge_agg = cfg.edge_agg
        self.next_node = cfg.next_node
        self.edge_rank = None

    def set_inst(self, inst):
        self.inst = inst
        self.rank_edges()

    def rank_edges(self):
        self.edge_rank = []
        for obj in self.inst["dists"]:
            obj_flatten = obj.flatten()
            obj_flatten = rankdata(obj_flatten, method="dense")
            obj = obj_flatten.reshape(obj.shape)
            self.edge_rank.append(obj)
        self.edge_rank = np.array(self.edge_rank)

        if self.edge_agg == "mean":
            self.edge_rank = np.mean(self.edge_rank, axis=0)
        elif self.edge_agg == "max":
            self.edge_rank = np.max(self.edge_rank, axis=0)
        elif self.edge_agg == "min":
            self.edge_rank = np.min(self.edge_rank, axis=0)
        else:
            raise ValueError("Method must be either 'mean'/'max'/'min'")

    def score_nodes(self, layer):
        scores = []
        for nid, node in enumerate(layer):
            last_visit = node[-1]
            to_visit = [i for i, n in enumerate(node[:-1]) if n == 0]
            if len(to_visit):
                if self.next_node == "min":
                    score = np.min([self.edge_rank[last_visit][tv] for tv in to_visit])
                elif self.next_node == "max":
                    score = np.max([self.edge_rank[last_visit][tv] for tv in to_visit])
                else:
                    raise ValueError(
                        "Node selection must be either 'greedy' or 'robust'"
                    )
                scores.append(score)
            else:
                # Static score for all nodes as we cannot discriminate between them
                scores.append(1)

        return scores


class GTNodeScorer(NodeScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None

        exp_path = resource.pretrained / self.cfg.prob.prefix / self.cfg.prob.size
        exp_cfg = OmegaConf.load(exp_path / self.cfg.model.type / "config.yaml")
        model_path = exp_path / self.cfg.model.type / "best_model.pt"
        self.set_model(exp_cfg, model_path)

    def set_model(self, exp_cfg, model_path):
        from morbdd.model.tsp import ParetoNodePredictor

        self.model = ParetoNodePredictor(exp_cfg.model)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @torch.no_grad()
    def get_node_emb(self, node_feat, edge_feat):
        node_emb, edge_emb = self.model.graph_tokenizer(node_feat, edge_feat)
        node_emb = self.model.graph_encoder(node_emb, edge_emb)  # B x n_vars x d_emb
        return node_emb

    @torch.no_grad()
    def get_score(self, layer_id, node_emb, layer, n_vars=10):
        layer = torch.from_numpy(np.array(layer)).float()
        B = layer.shape[0]
        node_emb = node_emb.repeat(B, 1, 1)

        last_visit = layer[:, -1]
        visit_mask = layer[:, :-1]
        visit_mask[torch.arange(B), last_visit.long()] = 2
        visit_enc = self.model.visit_encoder(visit_mask.long())
        # B x d_emb
        node_visit = self.model.node_visit_encoder2(
            self.model.node_visit_encoder1((node_emb + visit_enc)).sum(1)
        )
        customer_enc = node_emb[torch.arange(B), last_visit.long()]
        l = torch.tensor(np.array([layer_id] * B))
        l_enc = self.model.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))

        preds = self.model.pareto_predictor(node_visit + customer_enc + l_enc)
        preds = torch.softmax(preds, dim=-1)
        preds = preds.cpu().numpy()
        return preds[:, -1]
