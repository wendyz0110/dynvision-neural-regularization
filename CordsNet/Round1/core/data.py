"""
Round 1 data on the GPU: the offline cache (prepare/build_cache.py) is loaded once,
and batches are index tensors into it. No image decoding during training.

Batches always come from a single recording day (one monkey, one day per batch).
"""
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Cache:
    """The whole Round 1 cache (images, z-scored neural data, labels) on one device.

    Row k of every array = stim_idx k + 1.
    """

    def __init__(self, cache_dir, device, areas=("V1", "V4", "IT")):
        cache_dir = Path(cache_dir)
        meta = np.load(cache_dir / "meta.npz")
        self.stim_idx = meta["stim_idx"]
        self.day = meta["day"]
        self.split = meta["split"]
        self.concept = meta["concept"]
        self.device = device
        self.images = torch.from_numpy(np.load(cache_dir / "images_uint8.npy")).to(device)          # uint8
        self.pseudo_label = torch.from_numpy(meta["pseudo_label"]).to(device)
        nz = np.load(cache_dir / "neural.npz")
        self.neural = {a: torch.from_numpy(nz[a]).to(device) for a in areas}                         # (N, 20, E)
        self.mean, self.std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    def images_for(self, rows):
        """ImageNet-normalized float images (B, 3, H, W) for row indices (LongTensor on device)."""
        return (self.images[rows].float() / 255.0 - self.mean) / self.std


def eval_batches(cache, splits, batch_size, seed):
    """Fixed evaluation batches, identical to epoch 0: per day (ascending), the images of
    `splits` sorted by stim_idx, shuffled with one generator shared across days, cut into
    full batches (partial batch dropped). Returns a list of (day, row_indices)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in sorted(set(cache.day.tolist())):
        stims = sorted(int(s) for s, dd, sp in zip(cache.stim_idx, cache.day, cache.split) if dd == d and sp in splits)
        stims = rng.permutation(stims).tolist()
        for i in range(0, len(stims) - batch_size + 1, batch_size):
            out.append((d, np.array(stims[i:i + batch_size]) - 1))
    return out


def train_batches(cache, split, batch_size, seed, epoch):
    """Training batches for one epoch: each day's images shuffled and cut into full batches
    (partial batch dropped), then the batches of all days shuffled together.
    Deterministic in (seed, epoch), so runs with the same seed see the same batches."""
    rng = np.random.default_rng([seed, epoch])
    out = []
    for d in sorted(set(cache.day.tolist())):
        rows = np.where((cache.day == d) & (cache.split == split))[0]
        rows = rng.permutation(rows)
        out += [(d, rows[i:i + batch_size]) for i in range(0, len(rows) - batch_size + 1, batch_size)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]
