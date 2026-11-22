"""
ImageNet training images for the Round 3 ImageNet CE.

Augmentation exactly as CordsNet's own ImageNet training (wmws2/cordsnet, utils.py,
load_dataset('imagenet', 'train')): RandomResizedCrop(224), RandomHorizontalFlip, ToImage,
ToDtype(float32, scale=True), RandAugment(), Normalize(ImageNet mean / std). True labels
(ImageFolder: synset folders sorted = CordsNet's class order, checked).

One ImageNet batch accompanies every THINGS batch. An epoch is defined by THINGS, so each
epoch draws its first `n_batches` batches from a fresh permutation of the ImageNet train set,
seeded by (seed, epoch): every run (control / students) sees the same ImageNet images in the
same order, and a resumed epoch sees the same images again. The augmentation draws are seeded
through the DataLoader generator (same seed), so they repeat too.
"""
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cordsnet_train_transform():
    return v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandAugment(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageNetTrain:
    def __init__(self, train_dir, wnids_json, batch_size, seed, num_workers=8):
        self.ds = ImageFolder(train_dir, transform=cordsnet_train_transform())
        wnids = [d["wnid"] for d in json.load(open(wnids_json))]
        assert self.ds.classes == wnids, "ImageNet train folders differ from CordsNet's class order"
        self.batch_size, self.seed, self.num_workers = batch_size, seed, num_workers

    def __len__(self):
        return len(self.ds)

    def epoch_loader(self, epoch, n_batches):
        """Iterator over `n_batches` batches (images (B, 3, 224, 224), labels (B,)) for this epoch."""
        g = torch.Generator().manual_seed(self.seed * 10_000 + epoch)
        idx = torch.randperm(len(self.ds), generator=g)[:n_batches * self.batch_size].tolist()
        return DataLoader(Subset(self.ds, idx), batch_size=self.batch_size, shuffle=False, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True,
                          generator=torch.Generator().manual_seed(self.seed * 10_000 + epoch),
                          persistent_workers=False, prefetch_factor=4 if self.num_workers > 0 else None)
