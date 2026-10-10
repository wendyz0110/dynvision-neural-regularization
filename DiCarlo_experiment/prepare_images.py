"""
Build the IMAGE cache: every Imagenette image, preprocessed once, as uint8.

Why
---
Without data augmentation, the tensor a model sees for an image is identical
every epoch, so decoding + resizing the JPEG 100 times is wasted work. This
script does it once for the whole dataset (official train/ AND val/ folders),
independent of any split and of any target model.

Output (in --out_dir)
---------------------
    images_uint8.npy   [N, 3, 224, 224] uint8, pixels after
                       Resize(256) -> CenterCrop(224), BEFORE ToTensor/Normalize
    manifest.json      row order and everything needed to use the cache:
                       paths[i]  = relative path (image ID) of row i,
                                   same strings as in split.json
                       labels[i] = ground-truth class index of row i
                       preprocessing + normalization constants

To get exactly what data_utils' eval transform produces, a consumer does
    x = images[rows] (uint8) -> float / 255 -> (x - mean) / std
which is bit-for-bit what ToTensor() + Normalize() do (--verify checks this).

The manifest is written LAST: a folder without manifest.json is incomplete.

Usage (from DiCarlo_experiment/):
    python prepare_images.py \
        --data_root /scratch/wz2723/dynvision/data/imagenette2-320 \
        --out_dir   /scratch/wz2723/dynvision/cache/imagenette_images
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from utils.data_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _discover_classes,
    _list_images,
    build_transforms,
)

CACHE_FORMAT_VERSION = 1
RESIZE = 256
CROP = 224

logger = logging.getLogger("prepare_images")


def paths_digest(paths: List[str]) -> str:
    """Short fingerprint of the row order; other caches store it to prove
    they were built from this exact image cache."""
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:16]


class UInt8ImageDataset(Dataset):
    """Same PIL steps as data_utils' eval transform, but stops at uint8."""

    def __init__(self, root: Path, rel_paths: List[str]):
        self.root = Path(root)
        self.rel_paths = rel_paths
        self.transform = T.Compose([
            T.Resize(RESIZE),
            T.CenterCrop(CROP),
            T.PILToTensor(),  # uint8 [3, H, W], no scaling
        ])

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, index: int):
        with Image.open(self.root / self.rel_paths[index]) as img:
            image = img.convert("RGB")  # some Imagenette files are grayscale/CMYK
        return index, self.transform(image)


def verify_rows(root: Path, rel_paths: List[str], images: np.ndarray, n: int, seed: int = 0):
    """Compare cached rows against data_utils' own eval transform."""
    eval_tf = build_transforms(augment=False)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(rel_paths), size=min(n, len(rel_paths)), replace=False)
    worst = 0.0
    for i in rows:
        with Image.open(root / rel_paths[i]) as img:
            reference = eval_tf(img.convert("RGB"))
        from_cache = (torch.from_numpy(np.array(images[i])).float() / 255.0 - mean) / std
        worst = max(worst, (reference - from_cache).abs().max().item())
    logger.info("Verify: %d random rows, max |difference| vs data_utils eval transform = %.3g", len(rows), worst)
    if worst > 1e-5:
        raise RuntimeError("Cached images do NOT match the eval transform.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Unpacked imagenette2-320 folder (contains train/ and val/).")
    parser.add_argument("--out_dir", required=True, help="Folder for images_uint8.npy + manifest.json.")
    parser.add_argument("--num_workers", type=int, default=10, help="Parallel JPEG decoders (burst A100 node has 12 CPUs).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--verify", type=int, default=50, help="Rows to check against data_utils' transform (0 = skip).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    array_path = out_dir / "images_uint8.npy"

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists. Use --overwrite to rebuild.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()  # mark as incomplete while rebuilding

    # Rows: official train/ folder, then official val/ folder, each sorted
    # (data_utils' helpers -> the same relative-path strings as split.json).
    classes = _discover_classes(root / "train")
    if _discover_classes(root / "val") != classes:
        raise RuntimeError("Class folders differ between train/ and val/.")
    items = _list_images(root, "train", classes) + _list_images(root, "val", classes)
    rel_paths = [p for p, _ in items]
    labels = [int(label) for _, label in items]
    n = len(items)
    logger.info("Found %d images (%d classes).", n, len(classes))

    images = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint8, shape=(n, 3, CROP, CROP))

    loader = DataLoader(
        UInt8ImageDataset(root, rel_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    start = time.time()
    done = 0
    for indices, batch in loader:
        images[indices.numpy()] = batch.numpy()
        done += len(indices)
        if done % (args.batch_size * 20) < args.batch_size or done == n:
            logger.info("  %d / %d images (%.0f s)", done, n, time.time() - start)
    images.flush()

    if args.verify > 0:
        verify_rows(root, rel_paths, images, args.verify)

    manifest = {
        "kind": "image_cache",
        "format_version": CACHE_FORMAT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(root.resolve()),
        "n_images": n,
        "classes": classes,
        "array_file": array_path.name,
        "array_shape": [n, 3, CROP, CROP],
        "array_dtype": "uint8",
        "preprocessing": f"Resize({RESIZE}) -> CenterCrop({CROP}) -> uint8 pixels (no augmentation)",
        "to_model_input": "x.float() / 255 -> (x - mean) / std  (== ToTensor + Normalize)",
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
        "paths_sha256_16": paths_digest(rel_paths),
        "paths": rel_paths,
        "labels": labels,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    size_gb = array_path.stat().st_size / 1e9
    logger.info("Done in %.0f s. %s = %.2f GB. Manifest: %s", time.time() - start, array_path, size_gb, manifest_path)


if __name__ == "__main__":
    main()