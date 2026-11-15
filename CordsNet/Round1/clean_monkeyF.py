"""
Make a corrected copy of monkey F's binned neural data: the V4 and IT files are swapped
back. The original files are only read, never modified.

Why: the preprocessing notebook (ThingsNeuralData_monkeyF.ipynb) cuts the 1024 sorted
channels into 0:512 -> "V1", 512:832 -> "V4", 832:1024 -> "IT", i.e. monkey N's area order.
The cut points are right for monkey F, but F's order is V1 -> IT -> V4:
  - TVSD paper (Results, first section): monkey F has 3 V4 arrays (192 electrodes) and
    5 IT arrays (320 electrodes); the "V4" file has 320 electrodes, the "IT" file 192.
  - Response timing (inspect_monkeyF.ipynb, time to half of the peak): "V4" file 85 ms,
    "IT" file 55 ms; monkey N: V4 45 ms, IT 85 ms.
So the data are fine; only the labels of the last two blocks are swapped.

Output folder (--out_dir), monkey N's file names and layout:
  monkeyF_day{d}_V1_binned.npz   copy of the V1 file          (20, trials, 512)
  monkeyF_day{d}_IT_binned.npz   copy of the original "V4" file (20, trials, 320)
  monkeyF_day{d}_V4_binned.npz   copy of the original "IT" file (20, trials, 192)
  cleaning_info.json             source folder, the swap and why, per-file shapes, checks

Arrays without a visual response are NOT dropped yet (rule to be agreed with the PI).

Usage (on Torch):
  python clean_monkeyF.py \
      --raw_dir /scratch/wz2723/data/TVSD/monkeyF \
      --out_dir /scratch/wz2723/data/TVSD/monkeyF_clean
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
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

MONKEY, DAYS = "monkeyF", [1, 2, 3, 4]
SOURCE = {"V1": "V1", "IT": "V4", "V4": "IT"}          # corrected area <- original file label
EXPECTED = {"V1": 512, "V4": 192, "IT": 320}            # electrodes per area (TVSD paper, 64 per array)

raw, out = Path(args.raw_dir).resolve(), Path(args.out_dir).resolve()
assert raw != out, "--out_dir must differ from --raw_dir (the original files are never modified)"
out.mkdir(parents=True, exist_ok=True)
if (out / "cleaning_info.json").exists() and not args.overwrite:
    raise SystemExit(f"{out} already has cleaned data; use --overwrite to redo")

info = {"monkey": MONKEY, "raw_dir": str(raw), "created": datetime.now().isoformat(timespec="seconds"),
        "swap": {"IT": "original V4 file (channels 512:832 of the sorted 1024)",
                 "V4": "original IT file (channels 832:1024 of the sorted 1024)",
                 "V1": "original V1 file (unchanged)"},
        "reason": "preprocessing notebook used monkey N's V1-V4-IT order; monkey F's order is V1-IT-V4 "
                  "(TVSD paper: F has 3 V4 arrays = 192 electrodes, 5 IT arrays = 320; IT responds later than V4)",
        "dropped_arrays": "none yet",
        "files": {}, "half_max_ms": {}}

t_ms = np.arange(20) * 10 + 5                            # bin centers
total = {a: np.zeros(20) for a in EXPECTED}
for day in DAYS:
    idx = {}
    for area, src in SOURCE.items():
        src_name, dst_name = f"{MONKEY}_day{day}_{src}_binned.npz", f"{MONKEY}_day{day}_{area}_binned.npz"
        d = np.load(raw / src_name)
        mua, idx[area] = d["binned_mua"], d["train_idx"]
        assert mua.shape[0] == 20 and mua.shape[2] == EXPECTED[area], \
            f"{src_name}: shape {mua.shape}, expected (20, trials, {EXPECTED[area]}) for {area}"
        total[area] += mua.astype(np.float64).sum(axis=(1, 2)) / (mua.shape[1] * mua.shape[2])
        shutil.copy2(raw / src_name, out / dst_name)
        info["files"][dst_name] = {"source": src_name, "shape": list(mua.shape)}
        print(f"day {day}: {src_name} -> {dst_name}  {mua.shape}")
    assert all(np.array_equal(idx[a], idx["V1"]) for a in idx), f"day {day}: train_idx differs between areas"

# check: after the swap, V1 < V4 < IT in response latency (IT from 60 ms on, after its early transient)
for area, c in total.items():
    c = (c - c[0]) / (c.max() - c[0])
    first = 6 if area == "IT" else 0
    info["half_max_ms"][area] = int(t_ms[first + np.argmax(c[first:] >= 0.5)])
print("time to half of the peak (ms):", info["half_max_ms"])
assert info["half_max_ms"]["V1"] <= info["half_max_ms"]["V4"] < info["half_max_ms"]["IT"], \
    "after the swap V4 should respond before IT; check the files"

# check the copies read back identical to their sources
for name, f in info["files"].items():
    assert np.array_equal(np.load(out / name)["binned_mua"], np.load(raw / f["source"])["binned_mua"]), name

with open(out / "cleaning_info.json", "w") as fh:
    json.dump(info, fh, indent=2)
print(f"Done: corrected data in {out}")
