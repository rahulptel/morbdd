import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self, in_dim=2, out_dim=4, hidden_dim=8):
        super(SetEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x1 = self.relu(self.linear1(x))
        # print(x1.shape)
        x1_agg = torch.sum(x1, axis=1)
        # print(x1_agg.shape)

        return self.relu(self.linear2(x1_agg))


class ParetoStatePredictor(nn.Module):
    def __init__(self, cfg):
        super(ParetoStatePredictor, self).__init__()
        self.cfg = cfg
        self.instance_encoder = SetEncoder(in_dim=self.cfg.ie.in_dim,
                                           out_dim=self.cfg.ie.out_dim,
                                           hidden_dim=self.cfg.ie.hidden_dim)
        self.context_encoder = SetEncoder(in_dim=self.cfg.ce.in_dim,
                                          out_dim=self.cfg.ce.out_dim,
                                          hidden_dim=self.cfg.ce.hidden_dim)
        self.parent_encoder = SetEncoder(in_dim=self.cfg.pe.in_dim,
                                         out_dim=self.cfg.pe.out_dim,
                                         hidden_dim=self.cfg.pe.hidden_dim)

        self.node_encoder = nn.ModuleList()
        self.node_encoder.append(nn.Linear(2, self.cfg.ne[0]))
        self.node_encoder.append(nn.ReLU())
        for i in range(1, len(self.cfg.ne)):
            self.node_encoder.append(nn.Linear(self.cfg.ne[i-1], self.cfg.ne[i]))
            self.node_encoder.append(nn.ReLU())
        self.node_encoder = nn.Sequential(*self.node_encoder)

        pred_in_dim = self.cfg.ie.out_dim + self.cfg.ce.out_dim + self.cfg.pe.out_dim + self.cfg.ne[-1]
        self.predictor = nn.Sequential(nn.Linear(pred_in_dim, 1),
                                       nn.Sigmoid())

    def forward(self, instf, vf, nf, pf):
        # print(instf.shape, vf.shape, nf.shape, pf.shape)
        # print(self.node_encoder)
        ie = self.instance_encoder(instf)
        # print(ie.shape)
        ve = self.context_encoder(vf)
        # print(ve.shape)
        pe = self.parent_encoder(pf)
        # print(pe.shape)
        ne = self.node_encoder(nf)
        # print(ne.shape)
        emb = torch.concat((ie, ve, ne, pe), axis=1)
        # print(emb.shape)

        return self.predictor(emb)
