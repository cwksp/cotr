import copy

import torch.nn as nn
import torch.nn.functional as F

from .models import make, register


@register('meanz')
class Meanz(nn.Module):

    def __init__(self, encoder, projector, predictor, variant='excl'):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})
        self.variant = variant

    def _forward_train(self, data):
        n, k = data.shape[:2]
        z = self.projector(self.encoder(data.view(n * k, *data.shape[2:])))
        p = self.predictor(z)
        z = z.view(n, k, -1)
        p = p.view(n, k, -1)
        if self.variant == 'incl':
            e = z.mean(dim=1, keepdim=True).expand(-1, k, -1)
        elif self.variant == 'excl':
            e = (z.sum(dim=1, keepdim=True).expand(-1, k, -1) - z) / (k - 1)
        loss = -(F.normalize(p, dim=-1)
                 * F.normalize(e, dim=-1).detach()).sum(dim=-1).mean()
        return loss

    def forward(self, x, mode):
        if mode == 'train':
            return self._forward_train(x)
        elif mode == 'feat':
            return self.encoder(x)
        elif mode == 'embs':
            ret = dict()
            ret['feat'] = self.encoder(x)
            ret['proj'] = self.projector(ret['feat'])
            ret['pred'] = self.predictor(ret['proj'])
            return ret


@register('transitor')
class Transitor(nn.Module):

    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})

        self.encoder_ = copy.deepcopy(self.encoder)
        self.projector_ = copy.deepcopy(self.projector)

    def _forward_train(self, data):
        data_noaug = data[:, -1, ...]
        data = data[:, :-1, ...].contiguous()

        n, k = data.shape[:2]
        z = self.projector(self.encoder(data.view(n * k, *data.shape[2:])))
        p = self.predictor(z)
        z = z.view(n, k, -1)
        p = p.view(n, k, -1)

        e_smp = z.mean(dim=1)
        e_pred = self.projector_(self.encoder_(data_noaug))
        loss_e = -(F.normalize(e_pred, dim=-1)
                   * F.normalize(e_smp, dim=-1).detach()).sum(dim=-1).mean()

        e_pred = e_pred.view(n, 1, -1).expand(-1, k, -1)
        loss_p = -(F.normalize(p, dim=-1)
                   * F.normalize(e_pred, dim=-1).detach()).sum(dim=-1).mean()

        return (loss_e + loss_p) * 0.5

    def forward(self, x, mode):
        if mode == 'train':
            return self._forward_train(x)
        elif mode == 'feat':
            return self.encoder(x)
        elif mode == 'embs':
            ret = dict()
            ret['feat'] = self.encoder(x)
            ret['proj'] = self.projector(ret['feat'])
            ret['pred'] = self.predictor(ret['proj'])
            return ret
