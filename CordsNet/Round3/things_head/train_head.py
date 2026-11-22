"""
Train the new THINGS classification head on the stored features (extract_features.py).

Only this head is trained; the CordsNet backbone and the ImageNet head are frozen (their
output is the fixed 512-d feature, computed once).

Head: Linear(512, C) (C = non-overlapping THINGS concepts), trained as a plain Linear with
AdamW weight decay (on the weights only, not the bias), then converted exactly to CordsNet's
weight-norm form (g = row norms, v = W) so it matches the ImageNet head's structure. (Weight
decay on a weight-normed layer would act differently: decaying the direction v barely changes
the function.)
Loss: CordsNet's CE over the 30 stored steps (70-99), weights logspace(-3, 0, 30) normalized
(core.losses.weighted_ce), true THINGS labels.
Learning rate: cosine from --lr to 0 over all steps (per batch), as in the main training; a
constant rate made training unstable once train accuracy reached 1 (first run).

One run per weight decay; every epoch the head is evaluated on train and val. Two selections
over (weight decay, epoch), both saved:
  top1   best val top-1 at the last step (99 = 200 ms)
  ce     lowest val weighted CE (steps 70-99)

Output (<out_dir>):
  history.csv           per weight decay and epoch: lr at the epoch's end; train loss; train / val
                        top-1, top-5 (last step); val CE; mean |W| per class
  best_head_<sel>.pt    state dict of the selected head in weight-norm form (keys as CordsNet's
                        classifier.3: parametrizations.weight.original0 = g, original1 = v, bias)
  results_<sel>.npz     selected head: W, b, weight_decay, epoch, selection; val / train top-1 and
                        top-5 at every stored step; val CE; classes, steps
  config.json           the arguments

Usage (on Torch, GPU job, see ../slurm/things_head.sbatch):
  python train_head.py
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROUND3_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND3_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR / "Round1"))

from core.losses import weighted_ce  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@torch.no_grad()
def topk_acc(head, feats, labels, chunk=512):
    """top-1 and top-5 at every stored step: two (T,) arrays."""
    hit1 = torch.zeros(feats.shape[1], device=feats.device)
    hit5 = torch.zeros(feats.shape[1], device=feats.device)
    for i in range(0, len(feats), chunk):
        top5 = head(feats[i:i + chunk]).topk(5, dim=-1).indices                 # (n, T, 5)
        y = labels[i:i + chunk, None, None]
        hit1 += (top5[..., 0:1] == y).squeeze(-1).sum(0)
        hit5 += (top5 == y).any(-1).sum(0)
    return (hit1 / len(feats)).cpu().numpy(), (hit5 / len(feats)).cpu().numpy()


@torch.no_grad()
def mean_ce(head, feats, labels, chunk=512):
    total = 0.0
    for i in range(0, len(feats), chunk):
        n = len(feats[i:i + chunk])
        total += float(weighted_ce(head(feats[i:i + chunk]), labels[i:i + chunk], 0, feats.shape[1])) * n
    return total / len(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="CordsNet/Round3/outputs/things_head/features.npz")
    ap.add_argument("--out_dir", default="CordsNet/Round3/outputs/things_head/cosine_ep40")
    ap.add_argument("--weight_decays", nargs="+", type=float, default=[0.0, 0.01, 0.1, 1.0])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2723)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    device = torch.device(args.device)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)

    d = np.load(resolve(args.features))
    classes = d["classes"]
    feats = {s: torch.from_numpy(d["features"][d["split"] == s]).to(device) for s in ("train", "val")}
    labels = {s: torch.from_numpy(d["label"][d["split"] == s].astype(np.int64)).to(device) for s in ("train", "val")}
    n_cls, n_steps = len(classes), feats["train"].shape[1]
    log(f"{n_cls} classes, {n_steps} steps; train {len(labels['train'])}, val {len(labels['val'])} images; "
        f"ImageNet head |W| per class {np.linalg.norm(d['imagenet_W'], axis=1).mean():.3f} (reference)")

    history = []
    best = {"top1": None, "ce": None}
    better = {"top1": lambda new, old: new["val_top1"] > old["val_top1"],
              "ce": lambda new, old: new["val_ce"] < old["val_ce"]}
    n = len(labels["train"])
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    for wd in args.weight_decays:
        t0 = time.time()
        torch.manual_seed(args.seed)
        head = torch.nn.Linear(feats["train"].shape[-1], n_cls).to(device)
        # weight decay on the weights only, not on the bias
        opt = torch.optim.AdamW([{"params": [head.weight], "weight_decay": wd},
                                 {"params": [head.bias], "weight_decay": 0.0}], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=0.0)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)               # same batches for every wd
        best_wd = {"top1": None, "ce": None}
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(n, generator=gen).to(device)
            total = 0.0
            for i in range(0, n, args.batch_size):
                idx = order[i:i + args.batch_size]
                loss = weighted_ce(head(feats["train"][idx]), labels["train"][idx], 0, n_steps)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(idx)
            tr1, tr5 = topk_acc(head, feats["train"], labels["train"])
            va1, va5 = topk_acc(head, feats["val"], labels["val"])
            row = {"weight_decay": wd, "epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": total / n,
                   "train_top1": tr1[-1], "train_top5": tr5[-1], "val_top1": va1[-1], "val_top5": va5[-1],
                   "val_ce": mean_ce(head, feats["val"], labels["val"]),
                   "W_norm": float(head.weight.norm(dim=1).mean())}
            history.append(row)
            for sel in best_wd:
                if best_wd[sel] is None or better[sel](row, best_wd[sel]):
                    best_wd[sel] = {**row, "W": head.weight.detach().clone(), "b": head.bias.detach().clone(),
                                    "val_top1_t": va1, "val_top5_t": va5, "train_top1_t": tr1, "train_top5_t": tr5}
        bt, bc = best_wd["top1"], best_wd["ce"]
        log(f"wd {wd:g}: best val top-1 {bt['val_top1']:.4f} (top-5 {bt['val_top5']:.4f}) at epoch {bt['epoch']}; "
            f"lowest val CE {bc['val_ce']:.3f} at epoch {bc['epoch']} (val top-1 {bc['val_top1']:.4f}); "
            f"last epoch val top-1 {va1[-1]:.4f}, train top-1 {tr1[-1]:.4f} ({time.time() - t0:.0f}s)")
        for sel in best:
            if best[sel] is None or better[sel](best_wd[sel], best[sel]):
                best[sel] = best_wd[sel]

    with open(out_dir / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    for sel, b in best.items():
        # selected head -> CordsNet's weight-norm form (same function, checked)
        fc = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(b["W"].shape[1], n_cls))
        with torch.no_grad():
            fc.parametrizations.weight.original0.copy_(b["W"].norm(dim=1, keepdim=True).cpu())
            fc.parametrizations.weight.original1.copy_(b["W"].cpu())
            fc.bias.copy_(b["b"].cpu())
        assert torch.allclose(fc.weight, b["W"].cpu(), atol=1e-6), "weight-norm conversion changed the weights"
        torch.save(fc.state_dict(), out_dir / f"best_head_{sel}.pt")
        np.savez(out_dir / f"results_{sel}.npz", W=b["W"].cpu().numpy(), b=b["b"].cpu().numpy(),
                 weight_decay=b["weight_decay"], epoch=b["epoch"], selection=sel, val_ce=b["val_ce"],
                 classes=classes, steps=d["steps"], val_top1_t=b["val_top1_t"], val_top5_t=b["val_top5_t"],
                 train_top1_t=b["train_top1_t"], train_top5_t=b["train_top5_t"])
        log(f"selected by {sel}: wd {b['weight_decay']:g}, epoch {b['epoch']}: val top-1 {b['val_top1']:.4f}, "
            f"top-5 {b['val_top5']:.4f}, val CE {b['val_ce']:.3f}; train top-1 {b['train_top1']:.4f}")
    log(f"chance: top-1 {1 / n_cls:.4f}, top-5 {5 / n_cls:.4f}")
    log(f"saved {out_dir}: history.csv, best_head_top1.pt, best_head_ce.pt, results_top1.npz, results_ce.npz")


if __name__ == "__main__":
    main()
