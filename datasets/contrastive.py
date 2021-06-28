import random
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .datasets import register


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@register('contrastive-wrapper')
class ContrastiveWrapper(Dataset):

    def __init__(self, dataset, size, use_blur=True, n_per=2):
        self.dataset = dataset
        compose = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if use_blur:
            compose.append(
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        compose.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset.data_mean, dataset.data_std),
        ])
        self.transform = transforms.Compose(compose)
        self.n_per = n_per
        self.noaug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset.data_mean, dataset.data_std),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        ret = [self.transform(x) for _ in range(self.n_per - 1)]
        ret.append(self.transform(x)) # or noaug
        return torch.stack(ret)
