import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import make, register


@register('cotr-net')
class CotrNet(nn.Module):

    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})

    def forward(self, x, key='feat'):
        if isinstance(key, str):
            key = [key]
        if 'pred' in key:
            lv = 2
        elif 'proj' in key:
            lv = 1
        elif 'feat' in key:
            lv = 0
        else:
            lv = -1

        dic = dict()
        if lv >= 0:
            x = self.encoder(x)
            dic['feat'] = x
        if lv >= 1:
            x = self.projector(x)
            dic['proj'] = x
        if lv >= 2:
            x = self.predictor(x)
            dic['pred'] = x

        ret = [dic[_] for _ in key]
        if len(ret) == 1:
            ret = ret[0]
        return ret


@register('identity-head')
def identity_head(**kwargs):
    return nn.Identity()


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
        nn.BatchNorm1d(out_dim, affine=False),
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


@register('linear-head')
class LinearHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


@register('cosine-head')
class CosineHead(nn.Module):

    def __init__(self, in_dim, out_dim, tau=10.0, tau_learnable=True):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if tau_learnable:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        else:
            self.tau = tau

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.proto, dim=-1)
        ret = torch.mm(x, p.t()) * self.tau
        return ret
