"""
Imagenette data pipeline for the DynVision alignment experiment.

What this module does
---------------------
1. Reads the official Imagenette 320px release (imagenette2-320.tgz), which
   unpacks to:

       <root>/train/<wnid>/*.JPEG   (~9.5k images, all seen by the target)
       <root>/val/<wnid>/*.JPEG     (~3.9k images, NEVER seen by the target)

2. Builds OUR three splits (strategy agreed for the first run):

       train : 80% of the official train/ folder   -> trains aligned + control
       val   : 20% of the official train/ folder   -> per-epoch monitoring only
       test  : 100% of the official val/ folder    -> evaluation pipeline only

   The 80/20 split is stratified by class (each class is split 80/20 on its
   own) and is fully determined by `split_seed`.

3. Saves the split as JSON (relative image paths + labels) so any run, and
   the evaluation script, can reload exactly the same images. Relative paths
   are used instead of integer indices because indices depend on how a
   machine happens to list a directory; a path always names the same image.

4. Applies the target model's preprocessing (from its DynVision config):

       eval  : Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
       train : Resize(256) -> CenterCrop(224) -> [random augmentations]
               -> ToTensor -> Normalize

   The augmentations are the same set the target was trained with, so the
   target's responses to augmented images stay in-distribution. They are
   applied ONCE per image in the loader; train_alignment.py then feeds that
   single tensor to target, aligned and control alike.

Labels follow torchvision/DynVision folder convention: wnid folders sorted
alphabetically -> class index 0..9. Labels are only used for monitoring
(gt_accuracy); CE supervision comes from the target's pseudo-labels.

Batches are dicts: {"images", "labels", "sample_ids"}, where sample_ids are
positions in the saved split list for that split (so they map back to file
paths via split.json).

Entry points
------------
build_dataloaders(cfg) -> {"train": DataLoader, "val": DataLoader}
    Used by train_alignment.py through `data.factory`.
build_test_loader(cfg, split=None) -> DataLoader
    For the evaluation pipeline.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

logger = logging.getLogger("alignment_training")

# ImageNet statistics, as in the target's config (data_statistics.imagenette).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

SPLIT_FORMAT_VERSION = 1

# An item is (path relative to the dataset root, class index).
Item = Tuple[str, int]


# =============================================================================
# Transforms
# =============================================================================

def build_transforms(*, augment: bool) -> T.Compose:
    """
    Target-model preprocessing, with optional training augmentation.

    Order mirrors DynVision's get_dataset(): PIL-level transforms first,
    then tensor conversion to [0, 1], then normalization.
    """
    steps: List[Any] = [
        T.Resize(256),
        T.CenterCrop(224),
    ]

    if augment:
        # Exactly the target's training augmentation (transform_presets ->
        # torch -> train -> imagenette in its config).
        steps += [
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]

    steps += [
        T.ToTensor(),  # PIL -> float tensor in [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)


# =============================================================================
# Dataset
# =============================================================================

class ImageListDataset(Dataset):
    """Images given as an explicit list of (relative path, label)."""

    def __init__(self, root: Path, items: Sequence[Item], transform):
        self.root = Path(root)
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rel_path, label = self.items[index]
        # Some Imagenette files are grayscale/CMYK; the model needs 3 channels.
        with Image.open(self.root / rel_path) as img:
            image = img.convert("RGB")
        return {
            "images": self.transform(image),
            "labels": label,
            "sample_ids": index,  # position in this split's list in split.json
        }


# =============================================================================
# Discovering images and building the split
# =============================================================================

def _discover_classes(split_dir: Path) -> List[str]:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Expected folder not found: {split_dir}\n"
            "data.root must point to the unpacked imagenette2-320 folder "
            "(the one containing train/ and val/)."
        )
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found in {split_dir}")
    return classes


def _list_images(root: Path, split_name: str, classes: Sequence[str]) -> List[Item]:
    """All images of one official folder, sorted for machine-independence."""
    items: List[Item] = []
    for label, wnid in enumerate(classes):
        class_dir = root / split_name / wnid
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        items += [(p.relative_to(root).as_posix(), label) for p in files]
    return items


def _stratified_split(
    items: Sequence[Item],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Item], List[Item]]:
    """Split each class separately so class proportions match in train/val."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = random.Random(seed)  # independent of the global RNG state
    by_class: Dict[int, List[Item]] = {}
    for item in items:  # items are already sorted -> deterministic
        by_class.setdefault(item[1], []).append(item)

    train, val = [], []
    for label in sorted(by_class):
        class_items = list(by_class[label])
        rng.shuffle(class_items)
        n_val = round(len(class_items) * val_ratio)
        val += class_items[:n_val]
        train += class_items[n_val:]

    return sorted(train), sorted(val)


def _class_counts(items: Sequence[Item]) -> Dict[str, int]:
    counts = Counter(label for _, label in items)
    return {str(k): counts[k] for k in sorted(counts)}


def create_split(root: Path, *, val_ratio: float, seed: int) -> Dict[str, Any]:
    """Build the train/val/test split from the unpacked imagenette2-320."""
    root = Path(root)
    classes = _discover_classes(root / "train")
    test_classes = _discover_classes(root / "val")
    if classes != test_classes:
        raise RuntimeError(
            "Class folders differ between train/ and val/:\n"
            f"train: {classes}\nval:   {test_classes}"
        )

    official_train = _list_images(root, "train", classes)
    official_val = _list_images(root, "val", classes)
    train, val = _stratified_split(official_train, val_ratio=val_ratio, seed=seed)

    return {
        "format_version": SPLIT_FORMAT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(root.resolve()),
        "split_seed": seed,
        "val_ratio": val_ratio,
        "strategy": (
            "train/val: stratified split of the official train/ folder "
            "(images the target model was trained on); "
            "test: the full official val/ folder (never seen by the target)."
        ),
        "classes": classes,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(official_val),
        },
        "class_counts": {
            "train": _class_counts(train),
            "val": _class_counts(val),
            "test": _class_counts(official_val),
        },
        # Lists of [relative_path, label]; sample_ids index into these.
        "train": [list(x) for x in train],
        "val": [list(x) for x in val],
        "test": [list(x) for x in official_val],
    }


def save_split(split: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)


def load_split(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    if split.get("format_version") != SPLIT_FORMAT_VERSION:
        raise ValueError(f"Unsupported split file format in {path}")
    return split


def _items(split: Mapping[str, Any], name: str) -> List[Item]:
    return [(str(p), int(label)) for p, label in split[name]]


def _check_files_exist(root: Path, split: Mapping[str, Any]) -> None:
    missing = [
        p
        for name in ("train", "val", "test")
        for p, _ in split[name]
        if not (root / p).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in the split file are missing "
            f"under {root}, e.g. {missing[:3]}"
        )


def _check_no_overlap(split: Mapping[str, Any]) -> None:
    sets = {name: {p for p, _ in split[name]} for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise RuntimeError(
                f"Split overlap between {a} and {b}: {len(shared)} images, "
                f"e.g. {sorted(shared)[:3]}"
            )


def build_or_load_split(cfg) -> Dict[str, Any]:
    """
    Reuse `data.split_file` if given; otherwise create a new split.

    Either way the split actually used is written to
    <runtime.output_dir>/data/split.json, so every run directory is
    self-contained and the evaluation pipeline can point at it.
    """
    if cfg.data.root is None:
        raise ValueError("Set data.root to the unpacked imagenette2-320 folder.")
    root = Path(cfg.data.root)

    if cfg.data.split_file:
        split = load_split(Path(cfg.data.split_file))
        logger.info("Loaded data split from %s", cfg.data.split_file)
    else:
        split = create_split(
            root,
            val_ratio=cfg.data.val_ratio,
            seed=cfg.data.split_seed,
        )
        logger.info(
            "Created data split (seed=%d, val_ratio=%.2f)",
            cfg.data.split_seed,
            cfg.data.val_ratio,
        )

    _check_no_overlap(split)
    _check_files_exist(root, split)

    out_path = Path(cfg.runtime.output_dir) / "data" / "split.json"
    save_split(split, out_path)

    logger.info(
        "Split sizes: train=%d val=%d test=%d (%d classes). Saved to %s",
        split["counts"]["train"],
        split["counts"]["val"],
        split["counts"]["test"],
        len(split["classes"]),
        out_path,
    )
    return split


# =============================================================================
# DataLoaders
# =============================================================================

def _loader_kwargs(cfg) -> Dict[str, Any]:
    num_workers = int(cfg.data.num_workers)
    return {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }


def build_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Factory used by train_alignment.py (data.factory).

    train: shuffled, drop_last=True (no tiny last batch -> no inflated CKA),
           optional augmentation.
    val:   fixed order, no augmentation, same batch size as train so the
           per-epoch batch CKA stays comparable across epochs.
    """
    split = build_or_load_split(cfg)
    root = Path(cfg.data.root)

    train_set = ImageListDataset(
        root,
        _items(split, "train"),
        build_transforms(augment=bool(cfg.data.train_augmentation)),
    )
    val_set = ImageListDataset(
        root,
        _items(split, "val"),
        build_transforms(augment=False),
    )

    # Seeded generator -> reproducible shuffling AND reproducible augmentation
    # (DataLoader derives each worker's torch seed from it).
    generator = torch.Generator().manual_seed(int(cfg.runtime.seed))

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **_loader_kwargs(cfg),
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(cfg),
    )

    logger.info(
        "Train augmentation: %s | batch_size=%d | train batches/epoch=%d",
        bool(cfg.data.train_augmentation),
        int(cfg.data.batch_size),
        len(train_loader),
    )
    return {"train": train_loader, "val": val_loader}


def build_test_loader(
    cfg,
    split: Optional[Mapping[str, Any]] = None,
    *,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """
    Test loader for the evaluation pipeline: official val/ folder, fixed
    order, no augmentation. Pass the run's saved split (or set
    data.split_file) so evaluation uses exactly the training run's split.
    """
    if split is None:
        if not cfg.data.split_file:
            raise ValueError(
                "build_test_loader needs a split: pass `split` or set "
                "data.split_file to a run's data/split.json."
            )
        split = load_split(Path(cfg.data.split_file))

    test_set = ImageListDataset(
        Path(cfg.data.root),
        _items(split, "test"),
        build_transforms(augment=False),
    )
    kwargs = _loader_kwargs(cfg)
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    return DataLoader(test_set, shuffle=False, drop_last=False, **kwargs)