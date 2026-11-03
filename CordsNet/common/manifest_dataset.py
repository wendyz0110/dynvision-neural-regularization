"""
Dataset that reads images in the exact row order of a stimulus manifest CSV
(as written by prepare_images/build_things_manifest.py).

Item k of the dataset is row k of the (optionally split-filtered) manifest, so
model outputs collected with shuffle=False line up row by row with the manifest.
"""
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size=224):
    """Resize the whole image to image_size x image_size (no crop), then ImageNet-normalize.

    All TVSD/THINGS images are square, so this does not distort them, and the
    model sees the full image the monkey saw (CordsNet's own ImageNet eval
    pipeline instead does Resize(256) + CenterCrop(224)).
    """
    return v2.Compose([
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    """Images listed in a manifest CSV, in manifest order.

    Args:
        manifest_csv: path to the manifest (needs columns split, stim_idx, concept, image_path).
        splits: None for all rows, or a list such as ["train"] / ["test"] to keep only those
                rows (relative order is preserved).
        transform: image transform; defaults to build_transform().
        limit: if given, keep only the first `limit` rows (for quick smoke tests).
    """

    def __init__(self, manifest_csv, splits=None, transform=None, limit=None):
        with open(manifest_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if splits is not None:
            rows = [r for r in rows if r["split"] in set(splits)]
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"No rows selected from {manifest_csv} (splits={splits}, limit={limit})")

        self.rows = rows
        self.transform = transform if transform is not None else build_transform()
        self.split = [r["split"] for r in rows]
        self.stim_idx = [int(r["stim_idx"]) for r in rows]
        self.concept = [r["concept"] for r in rows]
        self.image_path = [r["image_path"] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        with Image.open(self.image_path[k]) as img:
            img = img.convert("RGB")
        return self.transform(img), k
