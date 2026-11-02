"""
Build the TVSD stimulus manifest: stimulus index -> THINGS image file.

Reads TVSD's `things_imgs.mat` (train_imgs / test_imgs) and writes one CSV row
per stimulus, in the exact order of the .mat file. That order is the order of
the columns of `train_MUA` / `test_MUA`, and `train_idx` in the binned
neural files is the 1-based position in this order.

Output columns
  split        "train" or "test"
  stim_idx     1-based index within the split (matches TVSD train_idx / test_idx)
  concept      THINGS concept name (e.g. "aardvark")
  things_path  path inside the THINGS image database, with "/" separators
  image_path   absolute path on disk (things_root / things_path)
  exists       whether image_path exists (only when --things_root is given)

Usage
  python build_things_manifest.py --mat things_imgs.mat --out things_manifest.csv \
      [--things_root /scratch/<netid>/data/THINGS/object_images]

Without --things_root the script only parses and checks the .mat file.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import h5py

N_TRAIN, N_TEST, N_CONCEPTS, IMGS_PER_CONCEPT = 22248, 100, 1854, 12


def read_strings(f, split, field):
    """Read a MATLAB cell array of strings stored in a v7.3 (HDF5) .mat file."""
    refs = f[split][field][()].flatten()
    return ["".join(map(chr, f[r][()].flatten())) for r in refs]


def load_split(f, split):
    concepts = read_strings(f, split, "class")
    paths = [p.replace("\\", "/") for p in read_strings(f, split, "things_path")]
    assert len(concepts) == len(paths), f"{split}: class / things_path length mismatch"
    return concepts, paths


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", required=True, help="path to TVSD things_imgs.mat")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--things_root", default=None,
                    help="folder that contains the concept folders (aardvark/, abacus/, ...)")
    args = ap.parse_args()

    with h5py.File(args.mat, "r") as f:
        tr_c, tr_p = load_split(f, "train_imgs")
        te_c, te_p = load_split(f, "test_imgs")

    root = Path(args.things_root) if args.things_root else None
    rows = []
    for split, concepts, paths in [("train", tr_c, tr_p), ("test", te_c, te_p)]:
        for i, (c, p) in enumerate(zip(concepts, paths), start=1):
            row = {"split": split, "stim_idx": i, "concept": c, "things_path": p}
            if root is not None:
                full = root / p
                row["image_path"] = str(full)
                row["exists"] = full.is_file()
            rows.append(row)

    print("== Checks on things_imgs.mat ==")
    ok = True
    counts = Counter(tr_c)
    ok &= check("train size", len(tr_p) == N_TRAIN, f"{len(tr_p)}")
    ok &= check("test size", len(te_p) == N_TEST, f"{len(te_p)}")
    ok &= check("train paths unique", len(set(tr_p)) == len(tr_p))
    ok &= check("test paths unique", len(set(te_p)) == len(te_p))
    ok &= check("train concepts", len(counts) == N_CONCEPTS, f"{len(counts)}")
    ok &= check("12 train images per concept", set(counts.values()) == {IMGS_PER_CONCEPT},
                f"{sorted(set(counts.values()))}")
    ok &= check("no test image in train", not (set(te_p) & set(tr_p)))
    ok &= check("concept == folder name",
                all(p.split("/")[0] == c for p, c in zip(tr_p + te_p, tr_c + te_c)))

    if root is not None:
        print(f"== Checks on image files under {root} ==")
        missing = [r for r in rows if not r["exists"]]
        ok &= check("all images found", not missing, f"{len(rows) - len(missing)}/{len(rows)} found")
        for r in missing[:10]:
            print("   missing:", r["things_path"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

    if not ok:
        print("Some checks FAILED - do not use this manifest until they are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
