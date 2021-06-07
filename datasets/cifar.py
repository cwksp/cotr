import os.path as osp
import pickle
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .datasets import register


class CIFAR(Dataset):

    def __init__(self, root_path, filenames, label_key, augment):
        self.data = []
        self.label = []
        for filename in filenames:
            with open(osp.join(root_path, filename), 'rb') as f:
                dic = pickle.load(f, encoding='bytes')
            self.data.extend(dic[b'data'])
            self.label.extend(dic[label_key])
        self.data = np.concatenate(self.data, axis=0) \
            .reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = [Image.fromarray(_) for _ in self.data]
        self.n_classes = max(self.label) + 1

        compose = []
        if augment == 'crop':
            compose.append(transforms.RandomCrop(32, padding=4))
            compose.append(transforms.RandomHorizontalFlip())
        elif augment == 'resized':
            compose.append(transforms.RandomResizedCrop(32))
            compose.append(transforms.RandomHorizontalFlip())
        self.norm_sub = (0.4914, 0.4822, 0.4465)
        self.norm_div = (0.2023, 0.1994, 0.2010)
        compose.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.norm_sub, self.norm_div),
        ])
        self.transform = transforms.Compose(compose)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]


@register('cifar10')
def cifar10(root_path, split, augment=None):
    if split == 'train':
        filenames = [f'data_batch_{i + 1}' for i in range(5)]
    elif split == 'test':
        filenames = ['test_batch']
    return CIFAR(root_path, filenames, b'labels', augment)


@register('cifar100')
def cifar100(root_path, split, augment=None):
    if split == 'train':
        filenames = ['train']
    elif split == 'test':
        filenames = ['test']
    return CIFAR(root_path, filenames, b'fine_labels', augment)
