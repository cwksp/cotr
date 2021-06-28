import argparse
import math
import os
import os.path as osp
import copy
import socket
from PIL import Image

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

import datasets
import models
import utils


def train_step(onl_net, data, optimizer, tar_net, momentum):
    n = data.shape[0] # n raw-images
    data_onl = data[:, :2, ...].contiguous()
    data_tar = data[:, 2:, ...].contiguous()

    k_onl = data_onl.shape[1]
    z_onl = onl_net(data_onl.view(n * k_onl, *data_onl.shape[2:]),
                    key='proj').view(n, k_onl, -1)

    k_tar = data_tar.shape[1]
    with torch.no_grad():
        z_tar = tar_net(data_tar.view(n * k_tar, *data_tar.shape[2:]),
                        key='proj').view(n, k_tar, -1)

    method = 'excl'
    if method == 'simsiam':
        outp = onl_net.predictor(z_onl.view(n * k_onl, -1)).view(n, k_onl, -1)
        gt = z_tar.flip(1)
    elif method == 'h-exp':
        outp = z_onl
        gt = onl_net.predictor(
            z_tar.view(n * k_tar, -1)).view(n, k_tar, -1).flip(1)
    elif method == 'excl':
        outp = onl_net.predictor(z_onl.view(n * k_onl, -1)).view(n, k_onl, -1)
        gt = z_tar.mean(dim=1, keepdim=True).expand(-1, k_onl, -1)

    loss = -(F.normalize(outp, dim=-1)
             * F.normalize(gt, dim=-1)).sum(dim=-1).mean()

    # z = F.normalize(z.mean(dim=1, keepdim=True), dim=-1).expand(-1, 2, -1) ##

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for tp, op in zip(tar_net.parameters(), onl_net.parameters()):
        tp.data = tp.data * momentum + op.data * (1.0 - momentum)

    return {
        'loss': loss.item(),
    }


# def train_step_simsiam_raw(model, data, optimizer, *args):
#     # bs, n_per = data.shape[:2]
#     z0, p0 = model(data[:, 0, ...], key=['proj', 'pred'])
#     z1, p1 = model(data[:, 1, ...], key=['proj', 'pred'])

#     z0 = z0.detach()
#     z1 = z1.detach()
#     loss = -(F.cosine_similarity(p1, z0, dim=-1).mean()
#              + F.cosine_similarity(p0, z1, dim=-1).mean()) / 2.0

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     return {
#         'loss': loss.item(),
#     }


def log_temp_scalar(k, v, t):
    writer.add_scalar(k, v, global_step=t)
    wandb.log({k: v}, step=t)


def resume_train(resume_path):
    logger.info(f'Resume from: {resume_path}')
    checkpoint = torch.load(resume_path)
    model = models.make(checkpoint['model'], load_sd=True).cuda()
    if config['mulgpu']:
        model = nn.DataParallel(model)

    optimizer = utils.make_optimizer(
        model.parameters(), checkpoint['optimizer'], load_sd=True)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=checkpoint['max_epoch'])
    for g in optimizer.param_groups:
        g['lr'] = checkpoint['optimizer']['args']['lr']
    for _ in range(checkpoint['epoch']):
        lr_scheduler.step()
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, lr_scheduler, start_epoch


def main(config_):
    # ---- Environment setup ---- #
    global config, logger, writer
    config = config_
    resume_path = config.get('resume')
    save_dir = config['save_dir']
    logger, writer = utils.set_save_dir(save_dir, remove=(resume_path is None))
    with open(osp.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    os.environ['WANDB_NAME'] = config['exp_name']
    os.environ['WANDB_DIR'] = config['save_dir']
    if not config.get('wandb_upload', False):
        os.environ['WANDB_MODE'] = 'dryrun'
    _ = config['wandb']
    os.environ['WANDB_API_KEY'] = _['api_key']
    wandb.init(project=_['project'], entity=_['entity'], config=config)

    logger.info(f'Hostname: {socket.gethostname()}')
    # -------- #

    # ---- Dataset, model and optimizer ---- #
    train_dataset = datasets.make(config['train_dataset'])
    test_dataset = datasets.make(config['test_dataset'])
    n_classes = train_dataset.n_classes

    logger.info('Train dataset: {}, shape={}'.format(len(train_dataset),
        tuple(train_dataset[0][0].shape)))
    logger.info('Test dataset: {}, shape={}'.format(len(test_dataset),
        tuple(test_dataset[0][0].shape)))
    logger.info(f'Num classes: {n_classes}')

    model = models.make(config['model']).cuda()

    logger.info(f'Model: #params={utils.compute_num_params(model)}')
    logger.info('- Encoder #params={}, out_dim={}'.format(
        utils.compute_num_params(model.encoder),
        model.encoder.out_dim))
    logger.info('- Projector #params={}, out_dim={}'.format(
        utils.compute_num_params(model.projector),
        model.projector.out_dim))
    logger.info('- Predictor #params={}, out_dim={}'.format(
        utils.compute_num_params(model.predictor),
        model.predictor.out_dim))

    if config['mulgpu']:
        model = nn.DataParallel(model)

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    # -------- #

    # ---- Ready to train ---- #
    max_epoch = config['max_epoch']
    milestone_epoch = config.get('milestone_epoch', 1)
    start_epoch = 1

    num_workers = 8
    train_cotr_dataset = datasets.make(
        config['wrapper'], args={'dataset': train_dataset})
    train_cotr_loader = DataLoader(train_cotr_dataset, config['batch_size'],
                                   shuffle=True, drop_last=True,
                                   num_workers=num_workers, pin_memory=True)

    trainset_loader = DataLoader(train_dataset, config['batch_size'],
                                 num_workers=num_workers, pin_memory=True)
    testset_loader = DataLoader(test_dataset, config['batch_size'],
                                num_workers=num_workers, pin_memory=True)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)

    if resume_path is not None:
        model, optimizer, lr_scheduler, start_epoch = resume_train(resume_path)

    tar_net = copy.deepcopy(model)
    tar_net.train()
    tar_momentum = config['tar_momentum']

    epoch_timer = utils.EpochTimer(max_epoch)
    # -------- #

    for epoch in range(start_epoch, max_epoch + 1):
        log_text = f'Epoch {epoch}'

        # ---- Train ---- #
        model.train()

        _bak_transform = train_dataset.transform
        train_dataset.transform = lambda x: x

        log_temp_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        _ = ['loss']
        ave_scalars = {k: utils.Averager() for k in _}

        pbar = tqdm(train_cotr_loader, desc='train', leave=False)
        for data in pbar:
            data = data.cuda()
            _ = train_step(model, data, optimizer, tar_net, tar_momentum)
            for k, v in _.items():
                ave_scalars[k].add(v, len(data))
            if epoch == start_epoch:
                pbar.set_description(desc=f"train loss={_['loss']:.4f}")

        log_text += ', train:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            log_temp_scalar('train/' + k, v, epoch)

        lr_scheduler.step()

        train_dataset.transform = _bak_transform
        # -------- #

        if epoch % milestone_epoch == 0:
            # ---- Test ---- #
            model.eval()
            pbar = tqdm(trainset_loader, desc='eval-center', leave=False)
            with torch.no_grad():
                c_centers = [utils.Averager() for _ in range(n_classes)]
                for data, label in pbar:
                    data, label = data.cuda(), label.cuda()
                    outp = model(data, key='proj')
                    outp = F.normalize(outp, dim=-1)
                    for x, l in zip(outp, label):
                        c_centers[l].add(x)
                c_centers = torch.stack([_.item() for _ in c_centers])
                c_centers = F.normalize(c_centers, dim=-1)

            _ = ['loss', 'acc']
            ave_scalars = {k: utils.Averager() for k in _}

            pbar = tqdm(testset_loader, desc='eval-testset', leave=False)
            with torch.no_grad():
                for data, label in pbar:
                    data, label = data.cuda(), label.cuda()
                    outp = model(data, key='proj')
                    outp = F.normalize(outp, dim=-1)
                    logits = torch.mm(outp, c_centers.t())
                    _ = {
                        'loss': F.cross_entropy(logits, label).item(),
                        'acc': (torch.argmax(logits, dim=1) == label) \
                            .float().mean().item()
                    }
                    for k, v in _.items():
                        ave_scalars[k].add(v, len(data))

            log_text += ', test:'
            for k, v in ave_scalars.items():
                v = v.item()
                log_text += f' {k}={v:.4f}'
                log_temp_scalar('test/' + k, v, epoch)
            # -------- #

        # ---- Save and summary ---- #
        model_ = model.module if config['mulgpu'] else model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'max_epoch': max_epoch,
        }
        torch.save(checkpoint, osp.join(save_dir, 'epoch-last.pth'))

        log_text += ', {} {}/{}'.format(*epoch_timer.step())
        logger.info(log_text)

        writer.flush()
        # -------- #

    linear_eval(train_dataset, test_dataset, model_.encoder)


def linear_eval(train_dataset, test_dataset, model):
    # save_dir = config['save_dir']

    img_size = train_dataset[0][0].shape[1]
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.),
                                     ratio=(3. / 4, 4. / 3),
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_dataset.data_mean,
                             train_dataset.data_std),
    ])
    test_dataset.transform = transforms.Compose([
        transforms.Resize(int(img_size * (8. / 7)),
                              interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(test_dataset.data_mean,
                             test_dataset.data_std)
    ])

    model.eval()
    if config['mulgpu']:
        model = nn.DataParallel(model)

    linear = nn.Linear(model.out_dim, train_dataset.n_classes).cuda()
    linear.weight.data.normal_(mean=0., std=0.01)
    linear.bias.data.zero_()

    econfig = config['linear_eval']
    optimizer = utils.make_optimizer(linear.parameters(), econfig['optimizer'])
    # -------- #

    # ---- Ready to train ---- #
    max_epoch = econfig['max_epoch']
    milestone_epoch = econfig.get('milestone_epoch', 1)
    last_test_acc = 0.

    num_workers = 8
    train_loader = DataLoader(train_dataset, econfig['batch_size'],
                              shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, econfig['batch_size'],
                             num_workers=num_workers, pin_memory=True)

    epoch_timer = utils.EpochTimer(max_epoch)
    # -------- #

    def adjust_learning_rate(optimizer, epoch):
        lr = econfig['optimizer']['args']['lr']
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - 1) / max_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(1, max_epoch + 1):
        log_text = f'Linear eval epoch {epoch}'

        # ---- Train ---- #
        linear.train()
        adjust_learning_rate(optimizer, epoch)

        # log_temp_scalar('linear-eval_lr',
        #     optimizer.param_groups[0]['lr'], epoch)

        _ = ['loss', 'acc']
        ave_scalars = {k: utils.Averager() for k in _}

        pbar = tqdm(train_loader, desc='linear-eval/train', leave=False)
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                data = model(data)
            logits = linear(data)
            loss = F.cross_entropy(logits, label)
            acc = (torch.argmax(logits, dim=1) == label).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _ = {'loss': loss.item(), 'acc': acc}
            for k, v in _.items():
                ave_scalars[k].add(v, len(data))

        log_text += ', train:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            # log_temp_scalar('linear-eval_train/' + k, v, epoch)

        # -------- #

        if epoch % milestone_epoch == 0:
            # ---- Test ---- #
            linear.eval()
            _ = ['loss', 'acc']
            ave_scalars = {k: utils.Averager() for k in _}

            pbar = tqdm(test_loader, desc='linear-eval/test', leave=False)
            for data, label in pbar:
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    data = model(data)
                    logits = linear(data)
                loss = F.cross_entropy(logits, label)
                acc = (torch.argmax(logits, dim=1) == label) \
                    .float().mean().item()
                _ = {'loss': loss.item(), 'acc': acc}
                for k, v in _.items():
                    ave_scalars[k].add(v, len(data))

            log_text += ', test:'
            for k, v in ave_scalars.items():
                v = v.item()
                log_text += f' {k}={v:.4f}'
                # log_temp_scalar('linear-eval_test/' + k, v, epoch)

            last_test_acc = ave_scalars['acc'].item()
            # -------- #

        # ---- Summary and save ---- #
        log_text += ', {} {}/{}'.format(*epoch_timer.step())
        logger.info(log_text)

        # pth_file = linear.state_dict()
        # torch.save(pth_file, osp.join(save_dir, 'linear-sd_last.pth'))

        # writer.flush()

    wandb.summary['lin-acc'] = last_test_acc
    logger.info('Linear eval acc: {:.3f}'.format(last_test_acc))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_cotr.yaml')
    parser.add_argument('--load-root', default='../../data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', '-g', default='0')
    parser.add_argument('--wandb-upload', action='store_true')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    return args


def make_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def translate_load_root_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_load_root_(v)
            elif isinstance(v, str):
                d[k] = v.replace('${load_root}', args.load_root)

    translate_load_root_(config)

    if args.name is None:
        exp_name = '_' + osp.basename(args.config).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    config['exp_name'] = exp_name
    save_dir = osp.join(args.save_root, exp_name)
    config['save_dir'] = save_dir

    config['wandb_upload'] = args.wandb_upload

    if args.resume is not None:
        config['resume'] = args.resume

    config['mulgpu'] = (len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1)
    return config


if __name__ == '__main__':
    args = make_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(make_config(args))
