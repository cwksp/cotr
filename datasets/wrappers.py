import random
from PIL import ImageFilter

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .datasets import register


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@register('instance-aug')
class InstanceAug(Dataset):

    def __init__(self, dataset, img_size, use_blur=True, n_per=2,
                 append_rep=False):
        self.dataset = dataset
        compose = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if use_blur:
            compose.append(transforms.RandomApply([
                GaussianBlur([0.1, 2])], p=0.5))
        compose.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset.data_mean, dataset.data_std),
        ])
        self.transform = transforms.Compose(compose)
        self.n_per = n_per
        self.rep = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset.data_mean, dataset.data_std),
        ])
        self.append_rep = append_rep

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        ret = [self.transform(x) for _ in range(self.n_per)]
        if self.append_rep:
            ret.append(self.rep(x))
        return torch.stack(ret)


@register('class-shots')
class ClassShots(Dataset):

    def __init__(self, dataset, n_per, img_size=32, repeat=1):
        self.dataset = dataset
        self.n_per = n_per
        self.repeat = repeat
        self.n_cls = dataset.num_classes
        self.cls_samples = [[] for _ in range(self.n_cls)]
        for i in range(len(dataset)):
            self.cls_samples[dataset.label[i]].append(i)
        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset.data_mean, dataset.data_std),
        ])
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(img_size, scale=(0.2, 1)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(dataset.data_mean, dataset.data_std),
        # ])

    def __len__(self):
        return self.n_cls * self.repeat

    def __getitem__(self, idx):
        idx %= self.n_cls
        ids = np.random.choice(self.cls_samples[idx], self.n_per, replace=False)
        return torch.stack([self.transform(self.dataset[_][0]) for _ in ids])
