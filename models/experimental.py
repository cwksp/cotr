import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import make, register


@register('meanz')
class Meanz(nn.Module):

    def __init__(self, encoder, projector, predictor, split=False, excl=True):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})
        self.split = split
        self.excl = excl

    def _forward_train(self, data):
        n, k = data.shape[:2]
        if not self.split:
            z = self.projector(self.encoder(data.view(n * k, *data.shape[2:])))
            p = self.predictor(z)
            z = z.view(n, k, -1)
            p = p.view(n, k, -1)
        else:
            z = []
            p = []
            for i in range(k):
                zi = self.projector(self.encoder(data[:, i, ...]))
                pi = self.predictor(zi)
                z.append(zi)
                p.append(pi)
            z = torch.stack(z, dim=1)
            p = torch.stack(p, dim=1)

        if not self.excl:
            e = z.mean(dim=1, keepdim=True).expand(-1, k, -1)
        else:
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


@register('ex-eest')
class ExEest(nn.Module):

    def __init__(self, encoder, projector, predictor, k_onl=2):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})
        self.k_onl = k_onl

    def _forward_train(self, data):
        n = data.shape[0]
        k_onl = self.k_onl
        k_est = data.shape[1] - k_onl
        data_onl = data[:, :k_onl, ...].contiguous()
        data_est = data[:, k_onl:, ...].contiguous()

        x = data_onl.view(n * k_onl, *data_onl.shape[2:])
        x = self.predictor(self.projector(self.encoder(x)))
        p = x.view(n, k_onl, -1)

        with torch.no_grad():
            x = data_est.view(n * k_est, *data_est.shape[2:])
            x = self.projector(self.encoder(x))
            e = x.view(n, k_est, -1).mean(dim=1, keepdim=True)\
                .expand(-1, k_onl, -1)

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


@register('transistor')
class Transistor(nn.Module):

    def __init__(self, encoder, projector, predictor, k_onl=2):
        super().__init__()
        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        # self.predictor = make(predictor,
        #                       args={'in_dim': self.projector.out_dim})
        self.k_onl = k_onl

        self.predictor = nn.Sequential(
            copy.deepcopy(self.encoder),
            copy.deepcopy(self.projector),
        )
        self.predictor.out_dim = self.projector.out_dim

    def _forward_train(self, data):
        data_noaug = data[:, -1, ...]
        data = data[:, :-1, ...].contiguous()

        n = data.shape[0]
        k_onl = self.k_onl
        k_est = data.shape[1] - k_onl
        data_onl = data[:, :k_onl, ...].contiguous()
        data_est = data[:, k_onl:, ...].contiguous()

        x = data_onl.view(n * k_onl, *data_onl.shape[2:])
        x = self.projector(self.encoder(x))
        z = x.view(n, k_onl, -1)

        with torch.no_grad():
            x = data_est.view(n * k_est, *data_est.shape[2:])
            x = self.projector(self.encoder(x))
            e = x.view(n, k_est, -1).mean(dim=1)

        p = self.predictor(data_noaug)

        loss_p2e = -(F.normalize(p, dim=-1)
                     * F.normalize(e, dim=-1).detach()).sum(dim=-1).mean()

        p = p.view(n, 1, -1).expand(-1, k_onl, -1)
        loss_z2p = -(F.normalize(z, dim=-1)
                     * F.normalize(p, dim=-1).detach()).sum(dim=-1).mean()

        loss = loss_p2e + loss_z2p
        return {
            'loss': loss,
            'loss_p2e': loss_p2e.item(),
            'loss_z2p': loss_z2p.item(),
        }

    def forward(self, x, mode):
        if mode == 'train':
            return self._forward_train(x)
        elif mode == 'feat':
            return self.encoder(x)
        elif mode == 'embs':
            ret = dict()
            ret['feat'] = self.encoder(x)
            ret['proj'] = self.projector(ret['feat'])
            ret['pred'] = self.predictor(x)
            return ret


@register('simsiam-distill')
class SimsiamDistill(nn.Module):

    def __init__(self, encoder, projector, predictor, teacher_ckpt,
                 teacher_mode='train'):
        super().__init__()
        simsiam_model = make(torch.load(teacher_ckpt)['model'], load_sd=True)
        simsiam_model.cpu()
        self.teacher = nn.Sequential(
            simsiam_model.encoder,
            simsiam_model.projector,
        )
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_mode = teacher_mode

        self.encoder = make(encoder)
        self.projector = make(projector,
                              args={'in_dim': self.encoder.out_dim})
        self.predictor = make(predictor,
                              args={'in_dim': self.projector.out_dim})

    def _forward_train(self, data):
        n = data.shape[0]
        k = data.shape[1] // 2
        data_stu = data[:, :k, ...].contiguous()
        data_tea = data[:, k:, ...].contiguous()

        outp_lst = []
        for i in range(k):
            outp = self.encoder(data_stu[:, i, ...])
            outp = self.projector(outp)
            outp = self.predictor(outp)
            outp_lst.append(outp)
        outp = torch.stack(outp_lst, dim=1)

        teacher = self.teacher
        if self.teacher_mode == 'train':
            teacher.train()
        else:
            teacher.eval()
        with torch.no_grad():
            gt_lst = []
            for i in range(k):
                gt = teacher(data_tea[:, i, ...])
                gt_lst.append(gt)
            gt = torch.stack(gt_lst, dim=1)

        # gt = gt.mean(dim=1, keepdim=True).expand(-1, k, -1)
        loss = -(F.normalize(outp, dim=-1)
                 * F.normalize(gt, dim=-1).detach()).sum(dim=-1).mean()
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
