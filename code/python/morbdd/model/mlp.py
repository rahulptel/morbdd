import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, bias=True, ln_eps=1e-5, act="relu", dropout=0.0, normalize=False):
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
