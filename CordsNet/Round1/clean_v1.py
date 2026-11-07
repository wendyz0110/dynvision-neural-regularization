"""
Make a cleaned copy of one monkey's binned neural data: V1 without the channels
that carry no visual response; V4 and IT copied unchanged. The original files
are only read, never modified.

Monkey N: V1 channels 320-383 (one full 64-channel array) show no visual response
on any of the 4 days; 512 - 64 = 448 = the V1 channel count in the TVSD paper.
Dropping them was confirmed with the PI (2026-11-06).

Output folder (--out_dir), same file names as the input:
  monkeyN_day{d}_V1_binned.npz   binned_mua (20, trials, 448), train_idx, kept_channels
  monkeyN_day{d}_V4_binned.npz   unchanged copy
  monkeyN_day{d}_IT_binned.npz   unchanged copy
  cleaning_info.json             source folder, dropped channels, per-file shapes

kept_channels = original V1 channel index (0-511) of each remaining column.

Usage (on Torch):
  python clean_v1.py \
      --raw_dir /scratch/wz2723/data/TVSD/monkeyN \
      --out_dir /scratch/wz2723/data/TVSD/monkeyN_clean
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--raw_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--monkey", default="monkeyN")
ap.add_argument("--drop_first", type=int, default=320, help="first V1 channel to drop (0-based)")
ap.add_argument("--drop_last", type=int, default=383, help="last V1 channel to drop (inclusive)")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

raw, out = Path(args.raw_dir).resolve(), Path(args.out_dir).resolve()
assert raw != out, "--out_dir must differ from --raw_dir (the original files are never modified)"
out.mkdir(parents=True, exist_ok=True)
if (out / "cleaning_info.json").exists() and not args.overwrite:
    raise SystemExit(f"{out} already has cleaned data; use --overwrite to redo")

drop = np.arange(args.drop_first, args.drop_last + 1)
info = {"monkey": args.monkey, "raw_dir": str(raw), "created": datetime.now().isoformat(timespec="seconds"),
        "v1_dropped_channels": [args.drop_first, args.drop_last], "files": {}}

for day in [1, 2, 3, 4]:
    # V1: drop the channels, keep the original index of every remaining column
    name = f"{args.monkey}_day{day}_V1_binned.npz"
    d = np.load(raw / name)
    mua, idx = d["binned_mua"], d["train_idx"]
    assert mua.shape[2] == 512, f"{name}: expected 512 V1 channels, got {mua.shape[2]}"
    kept = np.setdiff1d(np.arange(mua.shape[2]), drop)

    # sanity check: dropped channels should have no visual response (30-110 ms minus 0-20 ms)
    resp = mua[3:11].mean(axis=(0, 1)) - mua[:2].mean(axis=(0, 1))
    print(f"day {day}: |response| dropped channels max {np.abs(resp[drop]).max():.3f}, "
          f"kept channels median {np.median(resp[kept]):.3f}")

    np.savez_compressed(out / name, binned_mua=mua[:, :, kept], train_idx=idx, kept_channels=kept)
    info["files"][name] = {"shape_raw": list(mua.shape), "shape_clean": [mua.shape[0], mua.shape[1], len(kept)]}

    # V4, IT: unchanged copies, so everything Round 1 needs is in one folder
    for area in ["V4", "IT"]:
        name = f"{args.monkey}_day{day}_{area}_binned.npz"
        shutil.copy2(raw / name, out / name)
        info["files"][name] = {"shape_raw": list(np.load(raw / name)["binned_mua"].shape), "copied": True}

# check the cleaned V1 files read back correctly
for day in [1, 2, 3, 4]:
    c = np.load(out / f"{args.monkey}_day{day}_V1_binned.npz")
    r = np.load(raw / f"{args.monkey}_day{day}_V1_binned.npz")
    assert np.array_equal(c["train_idx"], r["train_idx"])
    assert np.array_equal(c["binned_mua"], r["binned_mua"][:, :, c["kept_channels"]])

with open(out / "cleaning_info.json", "w") as fh:
    json.dump(info, fh, indent=2)
print(f"Done: cleaned data in {out}")
