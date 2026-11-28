"""
Pack the selected Round 4 THINGS head into one self-contained file for
common/cordsnet_things.py (format: see that module's docstring).

Inputs (from train_head.py): best_head_<mode>_<sel>.pt (weight-norm state dict),
results_<mode>_<sel>.npz (class order, selection), history.csv (val metrics);
things_imagenet_overlap.csv (matched ImageNet classes).

Output: <out> (default CordsNet/Round4/outputs/things_head/things_head_1854.pt). The file is
checked by loading it back with load_things_head.

Usage (CPU, a few seconds; on Torch inside the container, see README / the sbatch comments):
  python export_head.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROUND4_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND4_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR))

from common.cordsnet_things import HEAD_FORMAT, load_things_head  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="CordsNet/Round4/outputs/things_head/cosine_ep40")
    ap.add_argument("--mode", default="all")
    ap.add_argument("--selection", default="top1")
    ap.add_argument("--overlap_csv", default="CordsNet/prepare_images/things_imagenet_overlap.csv")
    ap.add_argument("--out", default="CordsNet/Round4/outputs/things_head/things_head_1854.pt")
    args = ap.parse_args()
    run = resolve(args.run_dir)
    tag = f"{args.mode}_{args.selection}"

    sd = torch.load(run / f"best_head_{tag}.pt", map_location="cpu", weights_only=True)
    r = np.load(run / f"results_{tag}.npz")
    classes = [str(c) for c in r["classes"]]
    with open(resolve(args.overlap_csv), newline="") as fh:
        im_classes = {row["uniqueID"]: [int(i) for i in row["imagenet_idx"].split(";")] for row in csv.DictReader(fh)}
    is_overlap = [c in im_classes for c in classes]
    assert is_overlap == [bool(x) for x in r["is_overlap"]], "overlap flags differ from the training run"

    wd, epoch = float(r["weight_decay"]), int(r["epoch"])
    with open(run / "history.csv", newline="") as fh:
        hist = [row for row in csv.DictReader(fh)
                if row["mode"] == args.mode and float(row["weight_decay"]) == wd and int(row["epoch"]) == epoch]
    assert len(hist) == 1, "selected epoch not found in history.csv"
    val = {k: float(v) for k, v in hist[0].items() if k.startswith("val_")}

    head = {
        "format": HEAD_FORMAT,
        "state_dict": {k: v.clone() for k, v in sd.items()},
        "classes": classes,
        "is_overlap": is_overlap,
        "imagenet_classes": [im_classes.get(c, []) for c in classes],
        "description": (
            "THINGS head over all 1854 concepts for CordsNet-R8 (cordsnetr8.pth), on the 512-d classifier "
            "input (layer7 + layer8 -> global average pool). Trained on frozen pretrained features, steps 70-99 "
            "(142-200 ms), CordsNet's weighted CE, true THINGS labels; monkey N TVSD images, Round 1 split "
            "(train 10 / val 1 image per concept). Initialization: the 548 concepts that overlap ImageNet from "
            "the ImageNet head (one matched class: copied; several: mean), the 1306 others from the Round 3 head; "
            "then all rows trained (AdamW, weight decay on the weights only, lr 1e-3 cosine to 0, 40 epochs, "
            f"batch 256). Selected: weight decay {wd:g}, epoch {epoch}, by {args.selection} on val."),
        "training": {"mode": args.mode, "selection": args.selection, "weight_decay": wd, "epoch": epoch,
                     "lr": 1e-3, "epochs": 40, "batch_size": 256, "ce_steps": [70, 100],
                     "val": val, "source": str(run)},
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, out)
    back = load_things_head(out)
    assert back["classes"] == classes and torch.equal(back["state_dict"]["bias"], sd["bias"])
    print(f"saved {out}: {len(classes)} classes ({sum(is_overlap)} overlap ImageNet); weight decay {wd:g}, "
          f"epoch {epoch}; val top-1 {val['val_top1']:.4f}, val CE {val['val_ce']:.4f}")


if __name__ == "__main__":
    main()
