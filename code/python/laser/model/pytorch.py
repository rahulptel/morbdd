import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self, enc_net_dims=[2, 4], agg_net_dims=[4, 8]):
        super(SetEncoder, self).__init__()
        self.enc_net_dims = enc_net_dims
        self.agg_net_dims = agg_net_dims

        self.encoder_net = nn.ModuleList()
        for i in range(1, len(self.enc_net_dims)):
            self.encoder_net.append(nn.Linear(self.enc_net_dims[i - 1],
                                              self.enc_net_dims[i]))
            self.encoder_net.append(nn.ReLU())
        self.encoder_net = nn.Sequential(*self.encoder_net)

        self.aggregator_net = nn.ModuleList()
        for i in range(1, len(self.agg_net_dims)):
            self.aggregator_net.append(nn.Linear(self.agg_net_dims[i - 1],
                                                 self.agg_net_dims[i]))
            self.aggregator_net.append(nn.ReLU())
        self.aggregator_net = nn.Sequential(*self.aggregator_net)

    def forward(self, x):
        # print(x.shape)
        x_enc = self.encoder_net(x)
        # print(x1.shape)
        x_enc_agg = torch.sum(x_enc, axis=1)
        # print(x1_agg.shape)

        return self.aggregator_net(x_enc_agg)


class ParetoStatePredictor(nn.Module):
    def __init__(self, cfg):
        super(ParetoStatePredictor, self).__init__()
        self.cfg = cfg
        self.instance_encoder = SetEncoder(list(self.cfg.ie.enc), list(self.cfg.ie.agg))
        self.context_encoder = SetEncoder(list(self.cfg.ce.enc), list(self.cfg.ce.agg))
        self.parent_encoder = SetEncoder(list(self.cfg.pe.enc), list(self.cfg.pe.agg))

        self.node_encoder = nn.ModuleList()
        for i in range(1, len(self.cfg.ne)):
            self.node_encoder.append(nn.Linear(self.cfg.ne[i - 1], self.cfg.ne[i]))
            self.node_encoder.append(nn.ReLU())
        self.node_encoder = nn.Sequential(*self.node_encoder)

        pred_in_dim = self.cfg.ie.agg[-1] + self.cfg.ce.agg[-1] + self.cfg.pe.agg[-1] + self.cfg.ne[-1]
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
