import argparse
import os
import os.path as osp
import math
import copy
from PIL import Image

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import datasets
import models
import utils


def setup_wandb():
    os.environ['WANDB_NAME'] = cfg['exp_name']
    os.environ['WANDB_DIR'] = cfg['save_dir']
    if not cfg.get('wandb_upload', False):
        os.environ['WANDB_MODE'] = 'dryrun'
    wb = cfg['wandb']
    os.environ['WANDB_API_KEY'] = wb['api_key']
    wandb.init(project=wb['project'], entity=wb['entity'], config=cfg)


def log_temp_scalar(k, v, t):
    if is_master:
        writer.add_scalar(k, v, global_step=t)
        wandb.log({k: v}, step=t)


def train_step(model, data, optimizer):
    ret = model(data, mode='train')
    if isinstance(ret, dict):
        loss = ret.pop('loss')
    else:
        loss = ret
        ret = {}
    ret['loss'] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return ret


def dist_all_reduce_mean_(x):
    dist.barrier()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.div_(num_gpus)


def sync_ave_scalars_(ave_scalars):
    for k in ave_scalars.keys():
        x = torch.tensor([ave_scalars[k].item()]).cuda()
        dist_all_reduce_mean_(x)
        ave_scalars[k].v = x.mean().item()
        ave_scalars[k].n = 1


def train(model, data_loader, optimizer, epoch, log_text):
    model.train()

    raw_dataset = data_loader.dataset.dataset # wrapper's dataset
    _bak_transform = raw_dataset.transform
    raw_dataset.transform = transforms.Compose([]) # hack: no transform

    ##
    ave_scalars = {k: utils.Averager() for k in ['loss']}

    pbar = tqdm(data_loader, desc='train', leave=False)\
           if is_master else data_loader
    for data in pbar:
        data = data.cuda()
        _ = train_step(model, data, optimizer)
        for k, v in _.items():
            ave_scalars[k].add(v, len(data))
        if is_master:
            pbar.set_description(desc=f"train loss={_['loss']:.4f}")

    if distributed:
        sync_ave_scalars_(ave_scalars)

    log_text.append(', train:')
    for k, v in ave_scalars.items():
        v = v.item()
        log_text.append(f' {k}={v:.4f}')
        log_temp_scalar('train/' + k, v, epoch)

    raw_dataset.transform = _bak_transform # restore hack


def evaluate_centroid(model, trainset_loader, testset_loader, epoch,
                      log_text):
    num_classes = trainset_loader.dataset.num_classes

    model.eval()
    proto_keys = ['feat', 'proj', 'pred']

    pbar = tqdm(trainset_loader, desc='eval-center', leave=False)\
            if is_master else trainset_loader
    with torch.no_grad():
        protos = dict()
        for k in proto_keys:
            protos[k] = [utils.Averager() for _ in range(num_classes)]
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            embs = model(data, mode='embs')
            for k, emb in embs.items():
                for x, lb in zip(emb, label):
                    protos[k][lb.item()].add(x)
        for k in proto_keys:
            protos[k] = torch.stack([_.item() for _ in protos[k]])

    if distributed:
        for k in proto_keys:
            dist_all_reduce_mean_(protos[k])

    for k in proto_keys:
        protos[k] = F.normalize(protos[k], dim=-1)

    _ = ['acc_' + k for k in proto_keys]
    ave_scalars = {k: utils.Averager() for k in _}

    pbar = tqdm(testset_loader, desc='eval-testset', leave=False)\
           if is_master else testset_loader
    with torch.no_grad():
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            embs = model(data, mode='embs')
            for k, emb in embs.items():
                emb = F.normalize(emb, dim=-1)
                logits = torch.mm(emb, protos[k].t())
                acc = (torch.argmax(logits, dim=1) == label)\
                      .float().mean().item()
                ave_scalars['acc_' + k].add(acc, len(data))

    if distributed:
        sync_ave_scalars_(ave_scalars)

    log_text.append(', test:')
    for k, v in ave_scalars.items():
        v = v.item()
        log_text.append(f' {k}={v:.4f}')
        log_temp_scalar('test/' + k, v, epoch)


def save_checkpoint(model_sd, optimizer_sd, epoch, cfg_raw):
    model_spec = cfg['model']
    model_spec['sd'] = model_sd
    optimizer_spec = cfg['optimizer']
    optimizer_spec['sd'] = optimizer_sd
    checkpoint = {
        'model': model_spec,
        'optimizer': optimizer_spec,
        'epoch': epoch,
        'cfg': cfg_raw,
    }
    if is_master:
        torch.save(checkpoint, osp.join(cfg['save_dir'], 'epoch-last.pth'))


def main_worker(rank_, cfg_raw):
    # ---- Environment setup ---- #
    global cfg, log, writer
    global rank, num_gpus, distributed, is_master

    cfg = copy.deepcopy(cfg_raw)
    rank = rank_
    is_master = (rank == 0)
    num_gpus = cfg['num_gpus']
    distributed = (num_gpus > 1)

    eval_mode = (cfg.get('eval') is not None)
    if is_master:
        if eval_mode:
            log = print
        else:
            logger, writer = utils.set_save_dir(cfg['save_dir'], remove=False)
            with open(osp.join(cfg['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)
            log = logger.info
            setup_wandb()
    else:
        def empty_fn(*args, **kwargs):
            pass
        log = empty_fn

    if distributed:
        dist_url = f"tcp://localhost:{cfg['port']}"
        dist.init_process_group(backend='nccl', init_method=dist_url,
                                world_size=num_gpus, rank=rank)
        dist.barrier()
        log(f'Distributed training enabled.')

    cudnn.benchmark = cfg.get('cudnn', False)
    log(f'Env setup done.')
    # -------- #

    # ---- Dataset, model and optimizer ---- #
    train_dataset = datasets.make(cfg['train_dataset'])
    test_dataset = datasets.make(cfg['test_dataset'])

    log('Train dataset: {}, shape={}'.format(
        len(train_dataset), tuple(train_dataset[0][0].shape)))
    log('Test dataset: {}, shape={}'.format(
        len(test_dataset), tuple(test_dataset[0][0].shape)))
    log(f'Num classes: {train_dataset.num_classes}')

    if eval_mode:
        checkpoint = torch.load(cfg['eval'])
        model = models.make(checkpoint['model'], load_sd=True)
    else:
        model = models.make(cfg['model'])

    log(f'Model: #params={utils.compute_num_params(model)}')
    log('- Encoder #params={}, out_dim={}'.format(
        utils.compute_num_params(model.encoder), model.encoder.out_dim))
    log('- Projector #params={}, out_dim={}'.format(
        utils.compute_num_params(model.projector), model.projector.out_dim))
    log('- Predictor #params={}, out_dim={}'.format(
        utils.compute_num_params(model.predictor), model.predictor.out_dim))

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        if not eval_mode:
            cfg['batch_size'] = cfg['batch_size'] // num_gpus
        cfg['num_workers'] = cfg['num_workers'] // num_gpus
        ddp_model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model.cuda()
        ddp_model = model

    if eval_mode:
        evaluate_linear(train_dataset, test_dataset, ddp_model,
                        model.encoder.out_dim)
        return

    if cfg.get('fix_pred_lr', False):
        optim_params = [
            {'params': model.encoder.parameters(), 'fix_lr': False},
            {'params': model.projector.parameters(), 'fix_lr': False},
            {'params': model.predictor.parameters(), 'fix_lr': True},
        ]
    else:
        optim_params = ddp_model.parameters()
    optimizer = utils.make_optimizer(optim_params, cfg['optimizer'])
    # -------- #

    # ---- Ready to train ---- #
    max_epoch = cfg['max_epoch']
    eval_epoch = cfg.get('eval_epoch', 1)
    start_epoch = 1
    num_workers = cfg['num_workers']

    train_cotr_dataset = datasets.make(
        cfg['wrapper'], args={'dataset': train_dataset})
    train_cotr_sampler = DistributedSampler(train_cotr_dataset, drop_last=True)\
                         if distributed else None
    train_cotr_loader = DataLoader(
        train_cotr_dataset, cfg['batch_size'], drop_last=True,
        sampler=train_cotr_sampler, shuffle=(train_cotr_sampler is None),
        num_workers=num_workers, pin_memory=True)

    trainset_sampler = DistributedSampler(train_dataset)\
                       if distributed else None
    trainset_loader = DataLoader(
        train_dataset, cfg['eval_batch_size_tr'], sampler=trainset_sampler,
        num_workers=num_workers, pin_memory=True)

    testset_sampler = DistributedSampler(test_dataset)\
                      if distributed else None
    testset_loader = DataLoader(
        test_dataset, cfg['eval_batch_size_te'], sampler=testset_sampler,
        num_workers=num_workers, pin_memory=True)

    epoch_timer = utils.EpochTimer(max_epoch)
    # -------- #

    def adjust_learning_rate(optimizer, epoch, max_epoch):
        init_lr = cfg['optimizer']['args']['lr']
        lr = init_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max_epoch))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = lr

    for epoch in range(start_epoch, max_epoch + 1):
        log_text = [f'Epoch {epoch}']

        if distributed:
            train_cotr_sampler.set_epoch(epoch)
            trainset_sampler.set_epoch(epoch)
            testset_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, max_epoch)
        log_temp_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train(ddp_model, train_cotr_loader, optimizer, epoch, log_text)

        if epoch % eval_epoch == 0:
            evaluate_centroid(
                ddp_model, trainset_loader, testset_loader, epoch, log_text)

        save_checkpoint(model.state_dict(), optimizer.state_dict(), epoch,
                        cfg_raw)

        log_text.append(', {} {}/{}'.format(*epoch_timer.epoch_step()))
        log(''.join(log_text))
        if is_master:
            writer.flush()

    # last_acc = evaluate_linear(train_dataset, test_dataset, ddp_model,
    #                            model.encoder.out_dim)
    last_acc = -1
    if is_master:
        wandb.summary['lin-acc'] = last_acc
        writer.close()
        wandb.finish()


def evaluate_linear(train_dataset, test_dataset, ddp_model, feat_dim):
    ddp_model.eval()

    img_size = train_dataset[0][0].shape[1]
    train_compose = []
    test_compose = []
    if cfg['train_dataset']['name'].startswith('cifar'):
        train_compose.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1),
                                         ratio=(3 / 4, 4 / 3),
                                         interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ])
        test_compose.extend([
            transforms.Resize(round(img_size * (8 / 7)),
                              interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
        ])
    elif cfg['train_dataset']['name'].startswith('imagenet'):
        train_compose.extend([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ])
        test_compose.extend([
            transforms.Resize(round(img_size * (8 / 7))),
            transforms.CenterCrop(img_size),
        ])
    train_compose.extend([
        transforms.ToTensor(),
        transforms.Normalize(train_dataset.data_mean, train_dataset.data_std),
    ])
    test_compose.extend([
        transforms.ToTensor(),
        transforms.Normalize(test_dataset.data_mean, test_dataset.data_std),
    ])
    train_dataset.transform = transforms.Compose(train_compose)
    test_dataset.transform = transforms.Compose(test_compose)

    linear = nn.Linear(feat_dim, train_dataset.num_classes)
    linear.weight.data.normal_(mean=0, std=0.01)
    linear.bias.data.zero_()
    linear.cuda()
    if distributed:
        ddp_linear = DistributedDataParallel(linear, device_ids=[rank])
    else:
        ddp_linear = linear

    lcfg = cfg['linear_eval']
    lcfg['batch_size'] = lcfg['batch_size'] // num_gpus

    optimizer = utils.make_optimizer(ddp_linear.parameters(), lcfg['optimizer'])

    max_epoch = lcfg['max_epoch']
    eval_epoch = lcfg.get('eval_epoch', 1)
    last_acc = 0
    num_workers = cfg['num_workers']
    trainset_sampler = DistributedSampler(train_dataset, drop_last=True)\
                       if distributed else None
    trainset_loader = DataLoader(
        train_dataset, lcfg['batch_size'], drop_last=True,
        sampler=trainset_sampler, shuffle=(trainset_sampler is None),
        num_workers=num_workers, pin_memory=True)
    testset_sampler = None # no distributed sampler
    testset_loader = DataLoader(
        test_dataset, lcfg['eval_batch_size'], sampler=testset_sampler,
        num_workers=num_workers, pin_memory=True)
    epoch_timer = utils.EpochTimer(max_epoch)

    def adjust_learning_rate(optimizer, epoch):
        lr = lcfg['optimizer']['args']['lr']
        lr_scheduler = lcfg['lr_scheduler']
        if lr_scheduler == 'cosine':
            lr *= 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max_epoch))
        elif lr_scheduler == 'multistep':
            milestones = [round(max_epoch * 0.6), round(max_epoch * 0.8)]
            for milestone in milestones:
                lr *= 0.1 if epoch >= milestone else 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(1, max_epoch + 1):
        log_text = f'Linear eval epoch {epoch}'
        if distributed:
            trainset_sampler.set_epoch(epoch)
            if testset_sampler is not None:
                testset_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        ddp_linear.train()
        ave_scalars = {k: utils.Averager() for k in ['loss', 'acc']}
        pbar = tqdm(trainset_loader, desc='linear-eval/train', leave=False)\
               if is_master else trainset_loader
        for data, label in pbar:
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                data = ddp_model(data, mode='feat')
            logits = ddp_linear(data)
            loss = F.cross_entropy(logits, label)
            acc = (torch.argmax(logits, dim=1) == label).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _ = {'loss': loss.item(), 'acc': acc}
            for k, v in _.items():
                ave_scalars[k].add(v, len(data))
            logits = loss = None
        if distributed:
            sync_ave_scalars_(ave_scalars)
        log_text += ', train:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'

        if epoch % eval_epoch == 0:
            # ---- Test ---- #
            ddp_linear.eval()
            _ = ['loss', 'acc']
            ave_scalars = {k: utils.Averager() for k in _}
            pbar = tqdm(testset_loader, desc='linear-eval/test', leave=False)\
                   if is_master else testset_loader
            for data, label in pbar:
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    data = ddp_model(data, mode='feat')
                    logits = ddp_linear(data)
                    loss = F.cross_entropy(logits, label)
                acc = (torch.argmax(logits, dim=1) == label)\
                      .float().mean().item()
                _ = {'loss': loss.item(), 'acc': acc}
                for k, v in _.items():
                    ave_scalars[k].add(v, len(data))
            if distributed:
                sync_ave_scalars_(ave_scalars)
            log_text += ', test:'
            for k, v in ave_scalars.items():
                v = v.item()
                log_text += f' {k}={v:.4f}'
            last_acc = ave_scalars['acc'].item()

        log_text += ', {} {}/{}'.format(*epoch_timer.epoch_step())
        log(log_text)

    log('Linear eval acc: {:.3f}'.format(last_acc))
    return last_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/_.yaml')
    parser.add_argument('--load-root', default='../../data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', '-g', default=None)

    parser.add_argument('--port', '-p', default='29600')
    parser.add_argument('--wandb-upload', '-w', action='store_true')

    parser.add_argument('--eval', '-e', default=None)

    args = parser.parse_args()
    return args


def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('${load_root}', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = osp.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    cfg['exp_name'] = exp_name
    cfg['save_dir'] = osp.join(args.save_root, exp_name)
    cfg['num_gpus'] = torch.cuda.device_count()

    cfg['port'] = args.port
    cfg['wandb_upload'] = args.wandb_upload

    if args.eval is not None:
        cfg['eval'] = args.eval

    return cfg


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = make_cfg(args)
    if cfg.get('eval') is None:
        utils.ensure_path(cfg['save_dir'])
    if cfg['num_gpus'] > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=cfg['num_gpus'])
    else:
        main_worker(0, cfg)


if __name__ == '__main__':
    main()
