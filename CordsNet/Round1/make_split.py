"""
Split the 22,248 TVSD train images of one monkey into train / val / test,
per concept: of each concept's 12 images, 1 goes to test, 1 to val, 10 to train.

Each image was shown to the monkey once, on one recording day. The output CSV
records that day and the image's row in that day's npz files, so a batch can be
built from a single day and its neural data read directly.

Output CSV (one row per image, sorted by stim_idx):
  stim_idx    1-based TVSD train image index (= train_idx in the npz files,
              = stim_idx of the split == "train" rows in the stimulus manifest)
  concept     THINGS concept
  things_path image path relative to the THINGS object_images folder
  day         recording day (1-4) on which the image was shown
  day_row     0-based row of this image in that day's npz files (binned_mua[:, day_row, :])
  split       train / val / test

Usage (on Torch):
  python make_split.py \
      --neural_dir /scratch/wz2723/data/TVSD/monkeyN \
      --manifest /scratch/wz2723/data/THINGS/things_manifest_tvsd.csv \
      --out splits/monkeyN_split.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N_IMAGES, N_CONCEPTS, PER_CONCEPT = 22248, 1854, 12

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--neural_dir", required=True, help="folder with monkeyN_day{1-4}_{V1,V4,IT}_binned.npz")
ap.add_argument("--monkey", default="monkeyN")
ap.add_argument("--manifest", required=True, help="things_manifest_tvsd.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=2723)
ap.add_argument("--n_val", type=int, default=1, help="val images per concept")
ap.add_argument("--n_test", type=int, default=1, help="test images per concept")
args = ap.parse_args()

# ---- day and row of every image, from the neural files ----
# train_idx is identical across V1 / V4 / IT within a day; check that, then use it
day_of, row_of = {}, {}
for day in [1, 2, 3, 4]:
    idx = {a: np.load(Path(args.neural_dir) / f"{args.monkey}_day{day}_{a}_binned.npz")["train_idx"]
           for a in ["V1", "V4", "IT"]}
    assert np.array_equal(idx["V1"], idx["V4"]) and np.array_equal(idx["V1"], idx["IT"]), \
        f"day {day}: train_idx differs between areas"
    for row, s in enumerate(idx["IT"].tolist()):
        assert s not in day_of, f"stim_idx {s} appears on more than one day"
        day_of[s], row_of[s] = day, row
assert sorted(day_of) == list(range(1, N_IMAGES + 1)), "days do not cover stim_idx 1..22248 exactly once"

# ---- concept of every image, from the manifest ----
with open(args.manifest, newline="") as fh:
    train_rows = [r for r in csv.DictReader(fh) if r["split"] == "train"]
info = {int(r["stim_idx"]): r for r in train_rows}
assert sorted(info) == list(range(1, N_IMAGES + 1)), "manifest train rows do not cover stim_idx 1..22248"

by_concept = defaultdict(list)
for s in sorted(info):
    by_concept[info[s]["concept"]].append(s)
assert len(by_concept) == N_CONCEPTS and all(len(v) == PER_CONCEPT for v in by_concept.values()), \
    "expected 1854 concepts x 12 images"

# ---- per-concept split: first n_test of a random permutation -> test, next n_val -> val ----
rng = np.random.default_rng(args.seed)
split_of = {}
for c in sorted(by_concept):
    order = rng.permutation(by_concept[c]).tolist()
    for k, s in enumerate(order):
        split_of[s] = "test" if k < args.n_test else "val" if k < args.n_test + args.n_val else "train"

# ---- write ----
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["stim_idx", "concept", "things_path", "day", "day_row", "split"])
    for s in range(1, N_IMAGES + 1):
        w.writerow([s, info[s]["concept"], info[s]["things_path"], day_of[s], row_of[s], split_of[s]])

# ---- summary ----
counts = Counter((day_of[s], split_of[s]) for s in split_of)
print(f"Wrote {out} ({N_IMAGES} images, seed {args.seed})")
print(f"{'day':>5} {'train':>7} {'val':>6} {'test':>6} {'total':>7}")
for day in [1, 2, 3, 4]:
    tr, va, te = (counts[(day, sp)] for sp in ["train", "val", "test"])
    print(f"{day:>5} {tr:>7} {va:>6} {te:>6} {tr + va + te:>7}")
tot = Counter(split_of.values())
print(f"{'all':>5} {tot['train']:>7} {tot['val']:>6} {tot['test']:>6} {sum(tot.values()):>7}")
