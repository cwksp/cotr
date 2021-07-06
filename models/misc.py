import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register


@register('identity-head')
def identity_head(in_dim, **kwargs):
    net = nn.Identity()
    net.out_dim = in_dim
    return net


@register('linear-head')
class LinearHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.linear(x)
        return x


@register('cosine-head')
class CosineHead(nn.Module):

    def __init__(self, in_dim, out_dim, tau=10, tau_learnable=True):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if tau_learnable:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        else:
            self.tau = tau
        self.out_dim = out_dim

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.proto, dim=-1)
        ret = torch.mm(x, p.t()) * self.tau
        return ret


@register('simsiam-projector')
def simsiam_projector(in_dim, hid_dim=2048, out_dim=2048, n_layers=3):
    lst = [
        nn.Linear(in_dim, hid_dim, bias=False),
        nn.BatchNorm1d(hid_dim),
        nn.ReLU(inplace=True)
    ]
    for _ in range(n_layers - 2):
        lst.extend([
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        ])
    lst.extend([
        nn.Linear(hid_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False), # no bn-affine or relu at last
    ])
    net = nn.Sequential(*lst)
    net.out_dim = out_dim
    return net


@register('simsiam-predictor')
def simsiam_predictor(in_dim, hid_dim=512):
    net = nn.Sequential(
        nn.Linear(in_dim, hid_dim, bias=False),
        nn.BatchNorm1d(hid_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hid_dim, in_dim)
    )
    net.out_dim = in_dim
    return net
