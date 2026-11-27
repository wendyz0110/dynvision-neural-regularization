"""
Round 4 THINGS head: one Linear(512, 1854) over ALL THINGS concepts (548 that overlap ImageNet +
1306 that do not), trained on the stored features of the frozen pretrained model
(Round3/things_head/extract_features.py --concepts all).

Initialization (every row, before training):
  overlapping concept      mean of the effective weights (and biases) of its matched ImageNet
                           classes (things_imagenet_overlap.csv). 457 concepts match one class
                           (exact copy); 91 match several (e.g. dog: 118 breeds), where the mean
                           keeps the direction the classes share. A class shared by several
                           concepts (e.g. ambulance: ambulance, car) gives them identical rows.
  non-overlapping concept  the row of the Round 3 head trained on the same frozen features
                           (Round3/outputs/things_head/cosine_ep40/results_ce.npz).
Then the whole head is trained, because the two blocks come from different softmaxes (1000 vs
1306 classes) and their logit scales need not match in one 1854-way softmax.

Modes (each with the weight decay grid):
  all      every row trained
  frozen   the 548 overlapping rows kept at their initialization (restored after every
           optimizer step, so weight decay does not touch them either); the 1306 others trained

Training as for the Round 3 head (train_head.py there): CordsNet's weighted CE over steps 70-99,
true labels, AdamW (weight decay on the weights only), cosine lr per batch, same batches for
every run. Epoch 0 = the initialization, evaluated and selectable.

Metrics (val split; top-1 at the last step, 200 ms):
  val_top1 / val_top5 / val_ce            all 1854 concepts
  val_top1_overlap, val_top1_nonoverlap   images of overlapping / non-overlapping concepts
  val_top1_overlap_lenient                overlapping images; a prediction also counts as correct
                                          if the predicted concept shares an ImageNet class with the
                                          true one (car image -> "ambulance"), as in the ImageNet-head
                                          accuracy of Rounds 1-3
  W_norm_overlap, W_norm_nonoverlap       mean row norm of each block (shows weight decay shrinking
                                          the copied rows)
Reference values printed at the start, same val images: the ImageNet head on overlapping images
(Rounds 1-3 rule: correct if the predicted ImageNet class is one of the concept's classes), and
the Round 3 head (1306-way) on non-overlapping images.

Selections over (weight decay, epoch), per mode: top1 (best val top-1) and ce (lowest val CE).

Output (<out_dir>):
  history.csv                     per mode, weight decay, epoch: the metrics above + train loss/top-1
  best_head_<mode>_<sel>.pt       weight-norm state dict (keys as CordsNet's classifier.3)
  results_<mode>_<sel>.npz        W, b, weight_decay, epoch, val top-1 per step (all / overlap /
                                  non-overlap), classes, is_overlap, init_W, init_b
  config.json

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

ROUND4_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND4_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR / "Round1"))

from core.losses import weighted_ce  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@torch.no_grad()
def predictions(head, feats, chunk=512):
    """top-5 class indices at every stored step: (N, T, 5)."""
    return torch.cat([head(feats[i:i + chunk]).topk(5, dim=-1).indices for i in range(0, len(feats), chunk)])


@torch.no_grad()
def mean_ce(head, feats, labels, chunk=512):
    total = 0.0
    for i in range(0, len(feats), chunk):
        n = len(feats[i:i + chunk])
        total += float(weighted_ce(head(feats[i:i + chunk]), labels[i:i + chunk], 0, feats.shape[1])) * n
    return total / len(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="CordsNet/Round4/outputs/things_head/features.npz")
    ap.add_argument("--round3_head", default="CordsNet/Round3/outputs/things_head/cosine_ep40/results_ce.npz")
    ap.add_argument("--overlap_csv", default="CordsNet/prepare_images/things_imagenet_overlap.csv")
    ap.add_argument("--out_dir", default="CordsNet/Round4/outputs/things_head/cosine_ep40")
    ap.add_argument("--modes", nargs="+", default=["all", "frozen"], choices=["all", "frozen"])
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

    # ---- data ----
    d = np.load(resolve(args.features))
    assert "concepts" in d.files and str(d["concepts"]) == "all", "features must be extracted with --concepts all"
    classes = [str(c) for c in d["classes"]]
    n_cls = len(classes)
    feats = {s: torch.from_numpy(d["features"][d["split"] == s]).to(device) for s in ("train", "val")}
    labels = {s: torch.from_numpy(d["label"][d["split"] == s].astype(np.int64)).to(device) for s in ("train", "val")}
    n_steps = feats["train"].shape[1]

    # ---- overlap: concept -> matched ImageNet classes ----
    with open(resolve(args.overlap_csv), newline="") as fh:
        im_classes = {r["uniqueID"]: [int(i) for i in r["imagenet_idx"].split(";")] for r in csv.DictReader(fh)}
    is_ov_np = np.array([c in im_classes for c in classes])
    is_ov = torch.from_numpy(is_ov_np).to(device)
    ov_rows = torch.where(is_ov)[0]
    # equivalent[c, p]: predicting p for concept c counts as correct in the lenient accuracy
    sets = [set(im_classes.get(c, [])) for c in classes]
    equivalent = torch.eye(n_cls, dtype=torch.bool)
    for i in np.where(is_ov_np)[0]:
        for j in np.where(is_ov_np)[0]:
            if sets[i] & sets[j]:
                equivalent[i, j] = True
    equivalent = equivalent.to(device)
    val_ov = is_ov[labels["val"]]
    log(f"{n_cls} classes ({int(is_ov.sum())} overlap ImageNet, {int((~is_ov).sum())} not), {n_steps} steps; "
        f"train {len(labels['train'])}, val {len(labels['val'])} images ({int(val_ov.sum())} of overlapping concepts)")

    # ---- initialization ----
    W_im, b_im = torch.from_numpy(d["imagenet_W"]), torch.from_numpy(d["imagenet_b"])   # effective ImageNet head
    r3 = np.load(resolve(args.round3_head))
    r3_idx = {str(c): i for i, c in enumerate(r3["classes"])}
    W_r3, b_r3 = torch.from_numpy(r3["W"]), torch.from_numpy(r3["b"])
    W0, b0 = torch.zeros(n_cls, W_im.shape[1]), torch.zeros(n_cls)
    for i, c in enumerate(classes):
        if is_ov_np[i]:
            W0[i], b0[i] = W_im[im_classes[c]].mean(0), b_im[im_classes[c]].mean()
        else:
            assert c in r3_idx, f"non-overlapping concept {c} not in the Round 3 head"
            W0[i], b0[i] = W_r3[r3_idx[c]], b_r3[r3_idx[c]]
    n_multi = sum(len(im_classes[c]) > 1 for c in classes if c in im_classes)
    W0, b0 = W0.to(device), b0.to(device)
    log(f"init: {int(is_ov.sum()) - n_multi} overlapping rows copied, {n_multi} averaged over several ImageNet classes, "
        f"{int((~is_ov).sum())} from the Round 3 head; row norm: overlap {W0[is_ov].norm(dim=1).mean():.3f}, "
        f"non-overlap {W0[~is_ov].norm(dim=1).mean():.3f}")

    # ---- reference values on the same val images ----
    fv = feats["val"][:, -1]                                                       # last step
    y = labels["val"]
    im_pred = (fv @ W_im.to(device).T + b_im.to(device)).argmax(-1).cpu().numpy()
    hits = [im_pred[k] in im_classes[classes[int(y[k])]] for k in torch.where(val_ov)[0].tolist()]
    nonov_idx = torch.where(~val_ov)[0]
    r3_lab = torch.tensor([r3_idx[classes[int(c)]] for c in y[nonov_idx]], device=device)
    r3_pred = (fv[nonov_idx] @ W_r3.to(device).T + b_r3.to(device)).argmax(-1)
    log(f"reference: ImageNet head on overlapping val images (Rounds 1-3 rule) top-1 {np.mean(hits):.4f} "
        f"(n={len(hits)}); Round 3 head (1306-way) on non-overlapping val images top-1 "
        f"{float((r3_pred == r3_lab).float().mean()):.4f} (n={len(nonov_idx)})")

    def evaluate(head):
        p = predictions(head, feats["val"])                                        # (N, T, 5)
        hit1_t = p[..., 0] == y[:, None]                                           # (N, T)
        hit5 = (p[:, -1] == y[:, None]).any(-1)
        lenient = equivalent[y, p[:, -1, 0]]
        out = {"val_top1": float(hit1_t[:, -1].float().mean()), "val_top5": float(hit5.float().mean()),
               "val_ce": mean_ce(head, feats["val"], y),
               "val_top1_overlap": float(hit1_t[val_ov, -1].float().mean()),
               "val_top1_overlap_lenient": float(lenient[val_ov].float().mean()),
               "val_top1_nonoverlap": float(hit1_t[~val_ov, -1].float().mean()),
               "val_ce_overlap": mean_ce(head, feats["val"][val_ov], y[val_ov]),
               "val_ce_nonoverlap": mean_ce(head, feats["val"][~val_ov], y[~val_ov]),
               "W_norm_overlap": float(head.weight.detach()[is_ov].norm(dim=1).mean()),
               "W_norm_nonoverlap": float(head.weight.detach()[~is_ov].norm(dim=1).mean())}
        curves = {"val_top1_t": hit1_t.float().mean(0).cpu().numpy(),
                  "val_top1_overlap_t": hit1_t[val_ov].float().mean(0).cpu().numpy(),
                  "val_top1_nonoverlap_t": hit1_t[~val_ov].float().mean(0).cpu().numpy()}
        return out, curves

    # ---- training ----
    history, best = [], {}
    better = {"top1": lambda new, old: new["val_top1"] > old["val_top1"],
              "ce": lambda new, old: new["val_ce"] < old["val_ce"]}
    n = len(labels["train"])
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    for mode in args.modes:
        for wd in args.weight_decays:
            t0 = time.time()
            head = torch.nn.Linear(W0.shape[1], n_cls).to(device)
            with torch.no_grad():
                head.weight.copy_(W0)
                head.bias.copy_(b0)
            opt = torch.optim.AdamW([{"params": [head.weight], "weight_decay": wd},
                                     {"params": [head.bias], "weight_decay": 0.0}], lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=0.0)
            gen = torch.Generator(device="cpu").manual_seed(args.seed)             # same batches for every run
            for epoch in range(0, args.epochs + 1):
                total = float("nan")
                if epoch > 0:
                    order = torch.randperm(n, generator=gen).to(device)
                    total = 0.0
                    for i in range(0, n, args.batch_size):
                        idx = order[i:i + args.batch_size]
                        loss = weighted_ce(head(feats["train"][idx]), labels["train"][idx], 0, n_steps)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        sched.step()
                        if mode == "frozen":                                       # undo the step (and decay) on the copied rows
                            with torch.no_grad():
                                head.weight[ov_rows] = W0[ov_rows]
                                head.bias[ov_rows] = b0[ov_rows]
                        total += float(loss.detach()) * len(idx)
                    total /= n
                with torch.no_grad():
                    tr1 = float((predictions(head, feats["train"])[:, -1, 0] == labels["train"]).float().mean())
                m, curves = evaluate(head)
                row = {"mode": mode, "weight_decay": wd, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                       "train_loss": total, "train_top1": tr1, **m}
                history.append(row)
                for sel in better:
                    key = (mode, sel)
                    if key not in best or better[sel](row, best[key]):
                        best[key] = {**row, **curves, "W": head.weight.detach().clone(), "b": head.bias.detach().clone()}
                if epoch == 0 and wd == args.weight_decays[0]:
                    log(f"epoch 0 (initialization): val top-1 {m['val_top1']:.4f} (overlap {m['val_top1_overlap']:.4f}, "
                        f"lenient {m['val_top1_overlap_lenient']:.4f}; non-overlap {m['val_top1_nonoverlap']:.4f}), "
                        f"val CE {m['val_ce']:.3f}")
            log(f"{mode}, wd {wd:g}: last epoch val top-1 {row['val_top1']:.4f} (overlap {row['val_top1_overlap']:.4f}, "
                f"lenient {row['val_top1_overlap_lenient']:.4f}; non-overlap {row['val_top1_nonoverlap']:.4f}), "
                f"val CE {row['val_ce']:.3f}, train top-1 {tr1:.4f}; row norm overlap {row['W_norm_overlap']:.3f}, "
                f"non-overlap {row['W_norm_nonoverlap']:.3f} ({time.time() - t0:.0f}s)")

    with open(out_dir / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    for (mode, sel), b in best.items():
        # selected head -> CordsNet's weight-norm form (same function, checked)
        fc = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(b["W"].shape[1], n_cls))
        with torch.no_grad():
            fc.parametrizations.weight.original0.copy_(b["W"].norm(dim=1, keepdim=True).cpu())
            fc.parametrizations.weight.original1.copy_(b["W"].cpu())
            fc.bias.copy_(b["b"].cpu())
        assert torch.allclose(fc.weight, b["W"].cpu(), atol=1e-6), "weight-norm conversion changed the weights"
        torch.save(fc.state_dict(), out_dir / f"best_head_{mode}_{sel}.pt")
        np.savez(out_dir / f"results_{mode}_{sel}.npz", W=b["W"].cpu().numpy(), b=b["b"].cpu().numpy(),
                 weight_decay=b["weight_decay"], epoch=b["epoch"], mode=mode, selection=sel,
                 val_ce=b["val_ce"], val_top1=b["val_top1"], val_top1_t=b["val_top1_t"],
                 val_top1_overlap_t=b["val_top1_overlap_t"], val_top1_nonoverlap_t=b["val_top1_nonoverlap_t"],
                 classes=np.array(classes), is_overlap=is_ov_np, steps=d["steps"],
                 init_W=W0.cpu().numpy(), init_b=b0.cpu().numpy())
        log(f"selected ({mode}, by {sel}): wd {b['weight_decay']:g}, epoch {b['epoch']}: val top-1 {b['val_top1']:.4f} "
            f"(overlap {b['val_top1_overlap']:.4f}, lenient {b['val_top1_overlap_lenient']:.4f}; "
            f"non-overlap {b['val_top1_nonoverlap']:.4f}), top-5 {b['val_top5']:.4f}, val CE {b['val_ce']:.3f}")
    log(f"chance: top-1 {1 / n_cls:.4f}")
    log(f"saved {out_dir}: history.csv, best_head_<mode>_<sel>.pt, results_<mode>_<sel>.npz")


if __name__ == "__main__":
    main()
