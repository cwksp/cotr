import os
import os.path as osp
from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from .datasets import register


@register('imagenet')
class ImageNet(Dataset):

    def __init__(self, root_path, split, classes=None, transform=None,
                 img_size=224):
        root_path = osp.join(root_path, split)
        if classes is None:
            classes = sorted(os.listdir(root_path))
        self.class_names = classes
        self.num_classes = len(classes)
        self.data = []
        self.label = []
        for i, c in enumerate(classes):
            for x in os.listdir(osp.join(root_path, c)):
                self.data.append(osp.join(root_path, c, x))
                self.label.append(i)
        self.data_mean = (0.485, 0.456, 0.406)
        self.data_std = (0.229, 0.224, 0.225)
        compose = []
        if transform == 'resized-crop':
            compose.extend([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            compose.extend([
                transforms.Resize(round(img_size * (8 / 7))),
                transforms.CenterCrop(img_size),
            ])
        compose.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.data_mean, self.data_std),
        ])
        self.transform = transforms.Compose(compose)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as f:
            x = Image.open(f).convert('RGB')
        x = self.transform(x)
        return x, self.label[idx]

    def __len__(self):
        return len(self.data)


@register('imagenet100')
def imagenet100(root_path, **kwargs):
    with open(osp.join(root_path, 'imagenet100.txt'), 'r') as f:
        classes = [_.rstrip() for _ in f.readlines()]
    return ImageNet(root_path, classes=classes, **kwargs)
