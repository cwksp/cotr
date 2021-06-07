import argparse
import os
import os.path as osp
import socket
from torchvision.transforms.transforms import RandomResizedCrop

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

import datasets
import models
import utils


def train_step(model, data, label, optimizer):
    logits = model(data)
    loss = F.cross_entropy(logits, label)
    acc = (torch.argmax(logits, dim=1) == label).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'acc': acc.item(),
    }


def eval_step(model, data, label):
    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits, label)
        acc = (torch.argmax(logits, dim=1) == label).float().mean()

    return {
        'loss': loss.item(),
        'acc': acc.item(),
    }


def log_temp_scalar(k, v, t):
    writer.add_scalar(k, v, global_step=t)
    wandb.log({k: v}, step=t)


def main(config):
    # Environment setup
    resume = config.get('resume')
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    save_dir = config['save_dir']
    global logger, writer
    logger, writer = utils.set_save_dir(save_dir, remove=(resume is None))
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

    # Dataset, model and optimizer
    train_dataset = datasets.make(config['train_dataset'])
    test_dataset = datasets.make(config['test_dataset'])
    n_classes = train_dataset.n_classes
    logger.info('Train dataset: {}, shape={}'.format(len(train_dataset),
        tuple(train_dataset[0][0].shape)))
    logger.info('Test dataset: {}, shape={}'.format(len(test_dataset),
        tuple(test_dataset[0][0].shape)))
    logger.info(f'Num classes: {n_classes}')

    config['model']['args']['n_classes'] = n_classes
    model = models.make(config['model']).cuda()
    if n_gpus > 1:
        model = nn.DataParallel(model)
    logger.info(f'Model #params: {utils.compute_num_params(model)}')

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])

    # Ready to train
    max_epoch = config['max_epoch']
    n_milestones = config.get('n_milestones', 1)
    milestone_epoch = max_epoch // n_milestones
    min_test_loss = 1e18
    start_epoch = 1

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, config['batch_size'],
                             num_workers=4, pin_memory=True)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)

    if resume is not None:
        logger.info(f'Resume from: {resume}')
        pth_file = torch.load(resume)
        model = models.make(pth_file['model'], load_sd=True).cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)

        optimizer = utils.make_optimizer(
            model.parameters(), pth_file['optimizer'], load_sd=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)
        for g in optimizer.param_groups:
            g['lr'] = pth_file['optimizer']['args']['lr']
        for _ in range(pth_file['epoch']):
            lr_scheduler.step()
        start_epoch = pth_file['epoch'] + 1

    epoch_timer = utils.EpochTimer(max_epoch)

    for epoch in range(start_epoch, max_epoch + 1):
        log_text = f'Epoch {epoch}'

        # Train
        model.train()

        log_temp_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        _ = ['loss', 'acc']
        ave_scalars = {k: utils.Averager() for k in _}

        pbar = tqdm(train_loader, desc='train', leave=False)
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            _ = train_step(model, data, label, optimizer)
            for k, v in _.items():
                ave_scalars[k].add(v, len(data))
            pbar.set_description(desc=f"train loss:{_['loss']:.4f}")

        log_text += ', train:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            log_temp_scalar('train/' + k, v, epoch)

        # Test
        model.eval()

        _ = ['loss', 'acc']
        ave_scalars = {k: utils.Averager() for k in _}

        pbar = tqdm(test_loader, desc='test', leave=False)
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            _ = eval_step(model, data, label)
            for k, v in _.items():
                ave_scalars[k].add(v, len(data))
            pbar.set_description(desc=f"test loss:{_['loss']:.4f}")

        log_text += ', test:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            log_temp_scalar('test/' + k, v, epoch)

        test_loss = ave_scalars['loss'].item()
        lr_scheduler.step()

        # Summary and save
        log_text += ', {} {}/{}'.format(*epoch_timer.step())
        logger.info(log_text)

        model_ = model.module if n_gpus > 1 else model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        pth_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
        }

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(pth_file, osp.join(save_dir, 'min-test-loss.pth'))

        torch.save(pth_file, osp.join(save_dir, 'epoch-last.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--load-root', default='/data/cyb/data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', '-g', default='0')
    parser.add_argument('--wandb-upload', action='store_true')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

    main(config)
