import torch.nn as nn
import torch.nn.functional as F

from .models import make, register


@register('simsiam')
class Simsiam(nn.Module):

    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})

    def _forward_train(self, data):
        z0 = self.projector(self.encoder(data[:, 0, ...]))
        p0 = self.predictor(z0)
        z1 = self.projector(self.encoder(data[:, 1, ...]))
        p1 = self.predictor(z1)
        loss = -(F.cosine_similarity(p0, z1.detach(), dim=-1).mean()
                 + F.cosine_similarity(p1, z0.detach(), dim=-1).mean()) / 2
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
