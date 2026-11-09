"""
One-time offline cache for Round 1 training and evaluation, so the GPU jobs never
decode images or touch the raw neural files.

Everything is stored in stim_idx order (row k = stim_idx k + 1, the 22,248 TVSD train images):
  images_uint8.npy   (N, 3, 224, 224) uint8   images resized exactly as for the pseudo-labels
                                              (Resize 224x224, antialias, no crop); ImageNet
                                              normalization is applied later on the GPU
  neural.npz         V1 / V4 / IT  (N, 20, n_electrodes) float32, z-scored per day and electrode
                                   with the mean / SD of that day's train-split images, pooled
                                   over the 20 bins (same as epoch 0); V1 from the cleaned files
                     <area>_mean, <area>_sd  (4, n_electrodes) the z-score statistics per day
  meta.npz           stim_idx, day, day_row, split (train/val/test), concept, pseudo_label
  cache_info.json    sources, sha256 of the inputs, sizes

pseudo_label = argmax of the frozen pretrained CordsNet at the last stimulus step
(`pred_label` in the inference outputs.npz), for the TVSD train images.

Usage (on Torch, CPU job):
  python build_cache.py \
      --split_csv ../splits/monkeyN_split.csv \
      --manifest /scratch/wz2723/data/THINGS/things_manifest_tvsd.csv \
      --neural_dir /scratch/wz2723/data/TVSD/monkeyN_clean \
      --pseudo_labels /scratch/wz2723/cordsnet_runs/inference/things_cordsnetr8/outputs.npz \
      --out_dir /scratch/wz2723/data/TVSD/round1_cache_monkeyN
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CORDSNET_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORDSNET_DIR))
from common.manifest_dataset import ManifestDataset  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

N_BINS = 20
AREAS = ["V1", "V4", "IT"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Uint8Images(Dataset):
    """Manifest images as uint8 tensors, resized like build_transform() but before scaling."""

    def __init__(self, manifest_ds, rows, size):
        self.ds, self.rows = manifest_ds, rows
        self.tf = v2.Compose([v2.Resize((size, size), antialias=True), v2.ToImage()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        from PIL import Image
        with Image.open(self.ds.image_path[self.rows[k]]) as img:
            return self.tf(img.convert("RGB")), k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--neural_dir", required=True, help="cleaned neural folder (V1 without channels 320-383)")
    ap.add_argument("--monkey", default="monkeyN")
    ap.add_argument("--pseudo_labels", required=True, help="outputs.npz of the frozen-model inference run")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if (out / "cache_info.json").exists() and not args.overwrite:
        sys.exit(f"{out} already holds a cache; use --overwrite to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- split: one row per image, sorted by stim_idx ----
    with open(args.split_csv, newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: int(r["stim_idx"]))
    stim = np.array([int(r["stim_idx"]) for r in rows])
    n = len(stim)
    assert np.array_equal(stim, np.arange(1, n + 1)), "split CSV must cover stim_idx 1..N exactly once"
    day = np.array([int(r["day"]) for r in rows])
    day_row = np.array([int(r["day_row"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    concept = np.array([r["concept"] for r in rows])
    days = sorted(set(day.tolist()))
    print(f"{n} images; splits " + ", ".join(f"{s} {np.sum(split == s)}" for s in ["train", "val", "test"]))

    # ---- pseudo-labels from the inference run ----
    p = np.load(args.pseudo_labels)
    sel = p["split"] == "train"
    plabel = np.full(n, -1, dtype=np.int64)
    plabel[p["stim_idx"][sel] - 1] = p["pred_label"][sel]
    assert (plabel >= 0).all(), "pseudo-labels missing for some images"
    assert np.array_equal(p["concept"][sel][np.argsort(p["stim_idx"][sel])], concept), \
        "pseudo-label file and split CSV disagree on concepts"

    # ---- neural data: z-score per day and electrode, train-split stats pooled over bins ----
    neural, stats = {}, {}
    for area in AREAS:
        arr = None
        means, sds = [], []
        for d in days:
            f = np.load(Path(args.neural_dir) / f"{args.monkey}_day{d}_{area}_binned.npz")
            mua, idx = f["binned_mua"].astype(np.float32), f["train_idx"]
            assert mua.shape[0] == N_BINS
            on_day = np.where(day == d)[0]
            assert np.array_equal(np.sort(idx), np.sort(stim[on_day])), f"day {d} {area}: images differ from split CSV"
            assert np.array_equal(day_row[idx - 1], np.arange(len(idx))), f"day {d} {area}: day_row mismatch"
            train_rows = split[idx - 1] == "train"
            mean = mua[:, train_rows].mean(axis=(0, 1))
            sd = mua[:, train_rows].std(axis=(0, 1))
            assert (sd > 1e-6).all(), f"day {d} {area}: electrode with zero variance"
            if arr is None:
                arr = np.zeros((n, N_BINS, mua.shape[2]), dtype=np.float32)
            arr[idx - 1] = ((mua - mean) / sd).transpose(1, 0, 2)        # (trials, bins, electrodes)
            means.append(mean)
            sds.append(sd)
        neural[area] = arr
        stats[f"{area}_mean"], stats[f"{area}_sd"] = np.stack(means), np.stack(sds)
        print(f"neural {area}: {arr.shape}")
    np.savez(out / "neural.npz", **neural, **stats)

    # ---- images: uint8, resized like the pseudo-label inference ----
    mds = ManifestDataset(args.manifest, splits=["train"])
    pos = {s: k for k, s in enumerate(mds.stim_idx)}
    assert set(pos) >= set(stim.tolist()), "manifest is missing images"
    rows_in_manifest = [pos[s] for s in stim.tolist()]
    assert [mds.concept[k] for k in rows_in_manifest] == concept.tolist(), "manifest and split CSV disagree"
    images = np.lib.format.open_memmap(out / "images_uint8.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, 3, args.image_size, args.image_size))
    loader = DataLoader(Uint8Images(mds, rows_in_manifest, args.image_size), batch_size=64,
                        num_workers=args.num_workers)
    for b, (x, k) in enumerate(loader):
        images[k.numpy()] = x.numpy()
        if (b + 1) % 50 == 0:
            print(f"  images {int(k[-1]) + 1}/{n}  {(time.time() - t0) / 60:.1f} min", flush=True)
    images.flush()
    del images

    np.savez(out / "meta.npz", stim_idx=stim, day=day, day_row=day_row, split=split,
             concept=concept, pseudo_label=plabel)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_images": n, "image_size": args.image_size, "days": days,
        "normalization": "z-score per day and electrode; mean/SD over that day's train-split images and all 20 bins",
        "sources": {
            "split_csv": {"path": str(Path(args.split_csv).resolve()), "sha256": sha256(args.split_csv)},
            "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": sha256(args.manifest)},
            "pseudo_labels": {"path": str(Path(args.pseudo_labels).resolve()), "sha256": sha256(args.pseudo_labels)},
            "neural_dir": str(Path(args.neural_dir).resolve()),
        },
        "neural_shapes": {a: list(neural[a].shape) for a in AREAS},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(out / "cache_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Done in {info['minutes']} min: {out}")


if __name__ == "__main__":
    main()
