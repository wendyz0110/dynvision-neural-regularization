"""
Read training data from the image + target caches (prepare_images.py /
prepare_targets.py) instead of decoding JPEGs and running the target online.

One `CachedSplit` = one split (train or val) of the experiment:

    rows        : cache row of every image of the split, in split.json order
                  (matched by relative path = the image ID)
    pseudo      : target's final prediction, argmax(logits[:, label_t])
    gt          : ground-truth labels (monitoring only)
    targets     : {"IT_t19": memmap [N_cache, C, H, W], ...} -- only the
                  layers this run needs; rows are read from disk per batch

Batches come from a torch DataLoader whose workers read the memmaps in
parallel. Each batch is a dict of CPU tensors:

    images      uint8 [B, 3, 224, 224]  (normalize on the GPU: to_model_input)
    pseudo      int64 [B]
    gt          int64 [B]
    targets     {key: float16 [B, C, H, W]}
    sample_ids  int64 [B]  position in this split's list in split.json

Shuffling is deterministic per epoch: permutation = f(shuffle_seed, epoch).
Two runs with the same split, batch size and shuffle_seed therefore see
identical batches in identical order, and a resumed run continues with
exactly the batches it would have seen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def activation_key(layer: str, timestep: int) -> str:
    """Same file naming as prepare_targets.py: IT at t=19 -> 'IT_t19'."""
    return f"{layer}_t{timestep:02d}"


def _read_manifest(folder: Path, kind: str) -> Dict[str, Any]:
    path = Path(folder) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (cache missing or incomplete).")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("kind") != kind:
        raise ValueError(f"{path} is not a {kind} manifest.")
    return manifest


class CacheIndex:
    """Both manifests, checked against each other, plus path -> row lookup."""

    def __init__(self, image_dir: str | Path, target_dir: str | Path):
        self.image_dir = Path(image_dir)
        self.target_dir = Path(target_dir)
        self.image_manifest = _read_manifest(self.image_dir, "image_cache")
        self.target_manifest = _read_manifest(self.target_dir, "target_cache")

        im, tm = self.image_manifest, self.target_manifest
        if (tm["image_cache"]["paths_sha256_16"] != im["paths_sha256_16"]
                or tm["paths"] != im["paths"]):
            raise RuntimeError(
                f"Target cache {self.target_dir} was not built from image cache "
                f"{self.image_dir} (row order differs)."
            )
        self.paths: List[str] = im["paths"]
        self.labels = np.asarray(im["labels"], dtype=np.int64)
        self.row_of = {p: i for i, p in enumerate(self.paths)}
        self.mean = tuple(im["normalize_mean"])
        self.std = tuple(im["normalize_std"])

    @property
    def available(self) -> List[str]:
        return sorted(self.target_manifest["activations"])

    def require(self, keys: Sequence[str], purpose: str) -> None:
        missing = [k for k in keys if k not in self.target_manifest["activations"]]
        if missing:
            raise KeyError(
                f"Target cache {self.target_dir} has no {missing} (needed for {purpose}). "
                f"Available: {self.available}. Add them with prepare_targets.py "
                f"(--layers ... --timesteps ...) into the same --out_dir."
            )

    def fingerprint(self) -> Dict[str, Any]:
        """What a run records about the data it was trained on."""
        tm = self.target_manifest
        return {
            "image_cache": str(self.image_dir.resolve()),
            "target_cache": str(self.target_dir.resolve()),
            "paths_sha256_16": self.image_manifest["paths_sha256_16"],
            "target_checkpoint": tm["target"]["checkpoint"],
            "target_checkpoint_sha256_16": tm["target"]["checkpoint_sha256_16"],
            "target_precision": tm["target"]["precision"],
        }


class CachedSplit:
    """One split of the experiment, backed by the caches."""

    def __init__(
        self,
        index: CacheIndex,
        items: Sequence[Sequence[Any]],
        *,
        name: str,
        target_keys: Sequence[str],
        pseudo_label_timestep: int,
    ):
        self.index = index
        self.name = name
        self.target_keys = list(dict.fromkeys(target_keys))

        try:
            self.rows = np.asarray([index.row_of[str(p)] for p, _ in items], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"{name}: image {exc} from the split is not in the image cache.") from None

        split_labels = np.asarray([int(label) for _, label in items], dtype=np.int64)
        if not np.array_equal(split_labels, index.labels[self.rows]):
            raise RuntimeError(f"{name}: labels in split.json disagree with the image cache.")
        self.gt = split_labels

        logits = np.load(index.target_dir / index.target_manifest["logits"]["file"], mmap_mode="r")
        self.pseudo = np.asarray(logits[self.rows, pseudo_label_timestep]).argmax(-1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def pseudo_gt_agreement(self) -> float:
        return float((self.pseudo == self.gt).mean())


class _EpochBatchSampler(Sampler):
    """Yields arrays of split positions, one array per batch."""

    def __init__(self, n: int, batch_size: int, *, shuffle: bool, drop_last: bool, seed: int):
        self.n, self.batch_size = n, batch_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.n)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n, generator=g).numpy()

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else -(-self.n // self.batch_size)

    def __iter__(self) -> Iterator[np.ndarray]:
        order = self.order()
        for b in range(len(self)):
            # Sorted within the batch: faster disk reads; the order inside a
            # batch does not matter for CE or CKA (all tensors share it).
            yield np.sort(order[b * self.batch_size:(b + 1) * self.batch_size])


class _BatchReader(Dataset):
    """Reads one whole batch from the memmaps (opened lazily per worker)."""

    def __init__(self, split: CachedSplit):
        self.split = split
        self.image_file = split.index.image_dir / split.index.image_manifest["array_file"]
        acts = split.index.target_manifest["activations"]
        self.target_files = {k: split.index.target_dir / acts[k]["file"] for k in split.target_keys}
        self._images = None
        self._targets = None

    def __len__(self) -> int:
        return len(self.split)

    def _open(self) -> None:
        self._images = np.load(self.image_file, mmap_mode="r")
        self._targets = {k: np.load(f, mmap_mode="r") for k, f in self.target_files.items()}

    def __getitem__(self, positions: np.ndarray) -> Dict[str, Any]:
        if self._images is None:
            self._open()
        rows = self.split.rows[positions]
        # Cache rows are sorted because positions are sorted and split rows
        # are monotone in practice; either way fancy indexing is correct.
        return {
            "images": torch.from_numpy(np.asarray(self._images[rows])),
            "pseudo": torch.from_numpy(self.split.pseudo[positions]),
            "gt": torch.from_numpy(self.split.gt[positions]),
            "targets": {k: torch.from_numpy(np.asarray(m[rows])) for k, m in self._targets.items()},
            "sample_ids": torch.from_numpy(positions.astype(np.int64)),
        }


def read_batch(split: CachedSplit, positions: np.ndarray) -> Dict[str, Any]:
    """One batch (same dict as the loader yields) for given split positions."""
    return _BatchReader(split)[np.asarray(positions, dtype=np.int64)]


class CachedLoader:
    """Iterable over batches of a CachedSplit; call set_epoch(e) before each epoch."""

    def __init__(
        self,
        split: CachedSplit,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
        num_workers: int = 4,
    ):
        self.sampler = _EpochBatchSampler(len(split), batch_size, shuffle=shuffle, drop_last=drop_last, seed=seed)
        self.loader = DataLoader(
            _BatchReader(split),
            batch_size=None,          # the sampler already yields whole batches
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def order_digest(self) -> str:
        """Fingerprint of this epoch's sample order (compare aligned vs control)."""
        return hashlib.sha256(self.sampler.order().tobytes()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        return iter(self.loader)


def to_model_input(
    images_uint8: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    n_timesteps: int,
) -> torch.Tensor:
    """uint8 [B,3,H,W] on the device -> normalized [B,T,3,H,W] (same image every step).

    Identical to ToTensor() + Normalize() of data_utils' eval transform.
    """
    m = torch.tensor(mean, device=images_uint8.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=images_uint8.device).view(1, 3, 1, 1)
    x = (images_uint8.float() / 255.0 - m) / s
    return x.unsqueeze(1).expand(-1, n_timesteps, -1, -1, -1)