"""
Round 3: fine-tune CordsNet-R8 with two classification heads and CKA alignment (monkey N).

    loss = w_ce_imagenet * CE_ImageNet + w_ce_things * CE_THINGS
           + sum over alignment targets of weight * mean_k log(max(1 - CKA_S(layer, area, bin k), floor))

  CE_ImageNet  ImageNet train images (CordsNet's own augmentation), true labels, ImageNet head
               (logits 0-999); one ImageNet batch per THINGS batch (lib/imagenet_data.py).
  CE_THINGS    THINGS images of the 1306 concepts that do not overlap ImageNet, true labels,
               THINGS head (logits 1000-2305; initialized from Round3/things_head). Images of
               overlapping concepts get no CE (they still enter the CKA loss).
  CKA          as in Rounds 1-2 (Round1/core/losses.py), THINGS batches from one recording day.
  Both CEs: steps 70-99, weights logspace(-3, 0, 30) normalized (CordsNet's loss).

Each step: ImageNet batch forward + backward, then THINGS batch forward + backward (gradients add
up in .grad), then one optimizer step. Same gradient as summing the losses, but only one batch's
graph is in memory at a time. All parameters are trained (backbone and both heads).
Control: target weights 0 (CKA still computed and logged, without gradient); same seed -> same
THINGS and ImageNet batches in every run.

Evaluation at epoch 0 and after every epoch (lib/evaluate3.py): CKA with the Round 1-2 protocol
(val, batch 64), ImageNet head on THINGS (agreement, overlapping-concept top-1/5), THINGS head
on all non-overlapping val images (top-1 / top-5 at 200 ms, weighted CE). ImageNet-val accuracy
is evaluated afterwards on the saved checkpoints (Round3/imagenet/eval_imagenet.py).
Runs resume from <output_dir>/checkpoints/last.pt automatically.

Outputs (output_dir):
  model_card.yaml            full configuration, inputs with sha256, environment, progress
  config_used.yaml           the configuration after command-line overrides
  train_metrics.csv          one row per training step
  val_metrics.csv            one row per evaluation (epoch 0, 1, ...)
  eval/epoch_XXX.npz         evaluation arrays (Round 1 keys + things_head_*)
  checkpoints/epoch_XXX.pt   model weights every `save_every` epochs and at the last epoch
  checkpoints/last.pt        model + optimizer + scheduler + RNG state (every epoch), for resuming

Usage:
  python train.py --config ../configs/control.yaml
  python train.py --config ../configs/control.yaml --epochs 1 --max_steps 20 --output_dir <dir>   # pilot
"""
import argparse
import csv
import hashlib
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROUND3_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND3_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
for p in (CORDSNET_DIR, CORDSNET_DIR / "Round1", ROUND3_DIR):
    sys.path.insert(0, str(p))

from core.accuracy import load_overlap  # noqa: E402
from core.data import Cache, train_batches  # noqa: E402
from core.losses import as_float, cka_loss, cka_s_bins, parse_targets, weighted_ce  # noqa: E402
from core.recording import N_BINS, BinRecorder  # noqa: E402
from lib.evaluate3 import evaluate_round3  # noqa: E402
from lib.imagenet_data import ImageNetTrain  # noqa: E402
from lib.model import IMAGENET, THINGS, build_two_head_model  # noqa: E402
from lib.things_labels import things_labels  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def window_mean(arr, layer_i, area_j, bins, est):
    """Mean over a target's bins of a (layers, areas, bins, estimators) CKA array."""
    v = arr[layer_i, area_j, bins, est]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="stop each epoch after N steps (pilots)")
    ap.add_argument("--imagenet_dir", default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tr, lo, ev, im = cfg["train"], cfg["loss"], cfg["eval"], cfg["imagenet"]
    for key, sect in [("lr", tr), ("epochs", tr), ("batch_size", tr), ("max_steps", tr), ("device", cfg["runtime"])]:
        if getattr(args, key) is not None:
            sect[key] = getattr(args, key)
    if args.imagenet_dir is not None:
        im["train_dir"] = args.imagenet_dir
    if args.num_workers is not None:
        im["num_workers"] = args.num_workers
    out = Path(args.output_dir).resolve() if args.output_dir else resolve(cfg["runtime"]["output_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    device = torch.device(cfg["runtime"]["device"])
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731

    # ---- data ----
    t0 = time.time()
    cache = Cache(resolve(cfg["data"]["cache_dir"]), device, areas=ev["areas"])
    log(f"cache on {device} in {time.time() - t0:.0f}s: {len(cache.stim_idx)} images")

    # ---- model (two heads) ----
    mc = cfg["model"]
    ckpt_path, head_path, head_res = resolve(mc["checkpoint"]), resolve(mc["things_head"]), resolve(mc["things_head_results"])
    mk = {k: v for k, v in mc.items() if k not in ("checkpoint", "things_head", "things_head_results")}
    model, model_info, classes = build_two_head_model(ckpt_path, head_path, head_res, trainable=True, **mk)
    model = model.to(device).train()
    n_t = model_info["model_kwargs"]["n_timesteps"]
    assert n_t % N_BINS == 0, "stimulus steps must be a multiple of the 20 bins"
    spb, ce_steps = n_t // N_BINS, tuple(lo["ce_steps"])
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"model: {sum(p.numel() for p in params):,} trainable parameters; outputs {model_info['n_classes']} "
        f"(ImageNet 0-999, THINGS 1000-{model_info['n_classes'] - 1})")
    things_lab = things_labels(cache, classes)
    log(f"THINGS head labels: {int((things_lab >= 0).sum())} of {len(things_lab)} cache images "
        f"(train {int(((things_lab >= 0).cpu().numpy() & (cache.split == 'train')).sum())}); overlapping concepts get no THINGS CE")

    B, seed = tr["batch_size"], tr["seed"]
    steps_per_epoch = len(train_batches(cache, "train", B, seed, 1))
    if tr.get("max_steps"):
        steps_per_epoch = min(steps_per_epoch, tr["max_steps"])
    total_steps = steps_per_epoch * tr["epochs"]
    use_imnet = lo["w_ce_imagenet"] > 0
    imnet = None
    if use_imnet:
        imnet = ImageNetTrain(im["train_dir"], resolve(im["wnids_json"]), im.get("batch_size") or B, seed, im["num_workers"])
        log(f"ImageNet train: {len(imnet):,} images in {im['train_dir']}, batch {imnet.batch_size}, "
            f"{steps_per_epoch * imnet.batch_size:,} per epoch")
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    targets = parse_targets(lo)
    for t in targets:
        assert t["area"] in ev["areas"] and t["layer"] in ev["layers"], f"{t['tag']} not in eval areas / layers"
        t["li"], t["aj"] = ev["layers"].index(t["layer"]), ev["areas"].index(t["area"])
    student = any(t["weight"] > 0 for t in targets)
    rec_layers = list(dict.fromkeys(t["layer"] for t in targets))
    grad_bins = {name: sorted({k for t in targets if t["layer"] == name and t["weight"] > 0 for k in t["bins"]})
                 for name in rec_layers}
    overlap = load_overlap(resolve(ev["overlap_csv"]))
    log("targets: " + ", ".join(f"{t['tag']} bins {t['bins'][0]}-{t['bins'][-1]} weight {t['weight']}" for t in targets))

    # ---- resume ----
    start_epoch, global_step = 1, 0
    last = out / "checkpoints" / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["torch_rng"].cpu())   # map_location moved it to the GPU; the RNG state must be a CPU ByteTensor
        start_epoch, global_step = st["epoch"] + 1, st["global_step"]
        log(f"resumed from {last} (epoch {st['epoch']} done)")
        if start_epoch > tr["epochs"]:
            log("all epochs done")
            return
    else:
        torch.manual_seed(seed)

    # ---- model card ----
    card_path = out / "model_card.yaml"
    card = yaml.safe_load(open(card_path)) if card_path.exists() else {
        "run": out.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "description": ("student: CE_ImageNet + CE_THINGS + CKA_S alignment ("
                        + ", ".join(f"{t['tag']} x {t['weight']}" for t in targets) + ")"
                        if student else "control: CE_ImageNet + CE_THINGS (CKA logged, no gradient)"),
        "init_checkpoint": {"path": str(ckpt_path), "sha256": sha256(ckpt_path)},
        "things_head_init": {"path": str(head_path), "sha256": sha256(head_path), "classes": str(head_res)},
        "model": model_info,
        "data": {"cache_dir": str(resolve(cfg["data"]["cache_dir"])),
                 "train_split": "train", "train_images": int((cache.split == "train").sum()),
                 "batches": "one recording day per THINGS batch; per-epoch shuffle with seed (seed, epoch); partial batches dropped",
                 "things_augmentation": "none (CKA needs the exact images the monkey saw)",
                 "imagenet": {**im, "images": len(imnet) if imnet else 0,
                              "augmentation": "CordsNet: RandomResizedCrop(224), RandomHorizontalFlip, RandAugment, Normalize",
                              "sampling": "per epoch: first steps_per_epoch batches of a permutation seeded by (seed, epoch)"}},
        "loss": {**lo, "targets_parsed": [{k: t[k] for k in ("area", "layer", "weight", "tag")} | {"bins": [t["bins"][0], t["bins"][-1]]}
                                          for t in targets],
                 "ce": "weighted mean over outputs 70-99 (142-200 ms), weights logspace(-3,0,30)/sum, true labels; "
                       "ImageNet head = logits 0-999, THINGS head = logits 1000-2305; overlapping THINGS concepts: no CE",
                 "cka": "mean over valid bins of log(max(1 - CKA_S, floor)); CKA_S = Song et al. 2012",
                 "update": "ImageNet batch backward, THINGS batch backward, one optimizer step (gradients add up)"},
        "train": {**tr, "optimizer": "AdamW", "schedule": "cosine per step to 0", "steps_per_epoch": steps_per_epoch,
                  "total_steps": total_steps},
        "eval": ev,
        "environment": {"git_commit": git_commit(), "torch": str(torch.__version__),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "progress": [],
    }
    yaml.safe_dump(cfg, open(out / "config_used.yaml", "w"), sort_keys=False)

    val_path, train_path = out / "val_metrics.csv", out / "train_metrics.csv"
    # after a resume, drop rows / entries of the interrupted epoch (it is redone from its start)
    for path in [val_path, train_path]:
        if path.exists() and start_epoch > 1:
            with open(path, newline="") as fh:
                kept = [r for r in csv.DictReader(fh) if int(r["epoch"]) < start_epoch]
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(kept[0]) if kept else [])
                if kept:
                    w.writeheader()
                    w.writerows(kept)
            if not kept:
                path.unlink()
    card["progress"] = [p for p in card["progress"] if p["epoch"] < start_epoch]
    val_cols = (["epoch", "global_step", "things_head_top1", "things_head_top5", "things_head_ce",
                 "val_ce_pseudo", "agree", "imnet_head_things_top1", "imnet_head_things_top5"]
                + [f"CKA_S_{t['tag']}" for t in targets] + ["minutes"])
    train_cols = (["epoch", "step", "global_step", "day", "lr", "loss_total",
                   "loss_ce_imagenet", "imagenet_top1", "loss_ce_things", "things_top1", "n_things",
                   "grad_norm", "clipped", "seconds", "seconds_imagenet", "seconds_things"]
                  + [c for t in targets for c in [f"loss_cka_{t['tag']}", f"n_undefined_{t['tag']}", f"n_failed_{t['tag']}"]
                                                 + [f"cka_S_{t['tag']}_bin{k:02d}" for k in range(N_BINS)]])

    def run_eval(epoch):
        t = time.time()
        log(f"evaluating epoch {epoch} on {ev['splits']}")
        arrays, s = evaluate_round3(model, cache, things_lab, ev, n_t, ce_steps, overlap, log=log)
        np.savez(out / "eval" / f"epoch_{epoch:03d}.npz", **arrays)
        row = {"epoch": epoch, "global_step": global_step, "minutes": round((time.time() - t) / 60, 2)}
        row.update({k: round(s[k], 5) for k in ("things_head_top1", "things_head_top5", "things_head_ce",
                                                 "val_ce_pseudo", "agree", "imnet_head_things_top1", "imnet_head_things_top5")})
        for tg in targets:
            row[f"CKA_S_{tg['tag']}"] = round(window_mean(arrays["cka_pooled"], tg["li"], tg["aj"], tg["bins"], 1), 5)
        new = not val_path.exists()
        with open(val_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=val_cols)
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"epoch {epoch}: THINGS head top-1 {row['things_head_top1']:.4f} top-5 {row['things_head_top5']:.4f} "
            f"CE {row['things_head_ce']:.4f} (n = {s['things_head_n']}); ImageNet head on THINGS: agreement "
            f"{row['agree']:.4f}, overlap top-1 {row['imnet_head_things_top1']:.4f}; CKA_S: "
            + "  ".join(f"{tg['tag']} {row['CKA_S_' + tg['tag']]:.4f}" for tg in targets))
        return row

    if start_epoch == 1 and ev.get("at_start", True) and not (out / "eval" / "epoch_000.npz").exists():
        row = run_eval(0)
        card["progress"].append({"epoch": 0, **row})
        yaml.safe_dump(card, open(card_path, "w"), sort_keys=False)

    # ---- train ----
    for epoch in range(start_epoch, tr["epochs"] + 1):
        batches = train_batches(cache, "train", B, seed, epoch)[:steps_per_epoch]
        imnet_iter = iter(imnet.epoch_loader(epoch, len(batches))) if use_imnet else None
        # only each target's loss bins stay in the graph (none for weight 0); other bins are logged only
        rec = BinRecorder(model, rec_layers, grad_bins=grad_bins, steps_per_bin=spb)
        te = time.time()
        rows_out = []
        for step, (day, rows) in enumerate(batches, start=1):
            ts = time.time()
            opt.zero_grad(set_to_none=True)

            # 1. ImageNet batch: CE on the ImageNet head, backward now (frees its graph)
            loss_im, ce_im, im_top1 = None, float("nan"), float("nan")
            if use_imnet:
                xi, yi = next(imnet_iter)
                xi, yi = xi.to(device, non_blocking=True), yi.to(device, non_blocking=True)
                rec.enabled = False
                rec.reset()
                logits_i = model(xi.unsqueeze(1).expand(-1, n_t, -1, -1, -1))[..., IMAGENET]
                ce_i = weighted_ce(logits_i, yi, *ce_steps)
                loss_im = lo["w_ce_imagenet"] * ce_i
                loss_im.backward()
                ce_im = float(ce_i.detach())
                im_top1 = float((logits_i[:, -1].argmax(-1) == yi).float().mean())
                del logits_i, xi
                rec.enabled = True
            t_im = time.time() - ts

            # 2. THINGS batch (one day): CE on the THINGS head (non-overlapping concepts) + CKA
            rows_t = torch.as_tensor(rows, device=device)
            rec.reset()
            logits = model(cache.images_for(rows_t).unsqueeze(1).expand(-1, n_t, -1, -1, -1))
            rec.check_complete(n_t)
            lab = things_lab[rows_t]
            keep = lab >= 0
            loss = torch.zeros((), device=device)
            loss_th, th_top1 = None, float("nan")
            if keep.any():
                lt = logits[keep][..., THINGS]
                loss_th = weighted_ce(lt, lab[keep], *ce_steps)
                loss = loss + lo["w_ce_things"] * loss_th
                th_top1 = float((lt[:, -1].argmax(-1) == lab[keep]).float().mean())
            res = {}
            for tg in targets:
                neural = cache.neural[tg["area"]][rows_t]
                if tg["weight"] > 0:
                    vals, status = cka_s_bins(rec.bins[tg["layer"]], neural)
                    loss_cka = cka_loss(vals, tg["bins"], lo["floor"])
                else:
                    with torch.no_grad():
                        vals, status = cka_s_bins([None if b is None else b.detach() for b in rec.bins[tg["layer"]]], neural)
                        loss_cka = cka_loss(vals, tg["bins"], lo["floor"])
                if tg["weight"] > 0 and loss_cka is not None:
                    loss = loss + tg["weight"] * loss_cka
                res[tg["tag"]] = (vals, status, loss_cka)
            if loss.requires_grad:
                loss.backward()
            t_th = time.time() - ts - t_im

            # 3. one update with the summed gradients
            gn = float(torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"]))
            lr_used = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            global_step += 1

            total = float(loss.detach()) + (float(loss_im.detach()) if loss_im is not None else 0.0)
            row = {"epoch": epoch, "step": step, "global_step": global_step, "day": day, "lr": lr_used,
                   "loss_total": total, "loss_ce_imagenet": ce_im,
                   "imagenet_top1": im_top1, "loss_ce_things": as_float(loss_th), "things_top1": th_top1,
                   "n_things": int(keep.sum()), "grad_norm": gn, "clipped": int(gn > tr["grad_clip"]),
                   "seconds": round(time.time() - ts, 2), "seconds_imagenet": round(t_im, 2), "seconds_things": round(t_th, 2)}
            for tg in targets:
                vals, status, loss_cka = res[tg["tag"]]
                in_loss = [status[k] for k in tg["bins"]]
                row.update({f"loss_cka_{tg['tag']}": as_float(loss_cka),
                            f"n_undefined_{tg['tag']}": in_loss.count("undefined"),
                            f"n_failed_{tg['tag']}": in_loss.count("failed")})
                row.update({f"cka_S_{tg['tag']}_bin{k:02d}": as_float(v) for k, v in enumerate(vals)})
            rows_out.append(row)
            if step % tr["log_every"] == 0 or step == len(batches):
                new = not train_path.exists()
                with open(train_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=train_cols)
                    if new:
                        w.writeheader()
                    w.writerows(rows_out)
                rows_out = []
                el = time.time() - te
                log(f"epoch {epoch} step {step}/{len(batches)}  loss {row['loss_total']:.4f}  "
                    f"CE_ImageNet {row['loss_ce_imagenet']:.4f} (top-1 {row['imagenet_top1']:.2f})  "
                    f"CE_THINGS {row['loss_ce_things']:.4f}  "
                    + "  ".join(f"L_{tg['tag']} {row['loss_cka_' + tg['tag']]:.4f}" for tg in targets)
                    + f"  grad {gn:.1f}  {row['seconds']:.1f}s/step ({row['seconds_imagenet']:.1f} + {row['seconds_things']:.1f})  "
                    f"~{el / step * (len(batches) - step) / 60:.0f} min left in epoch")
        rec.detach()
        del imnet_iter
        train_minutes = (time.time() - te) / 60

        if epoch % tr["save_every"] == 0 or epoch == tr["epochs"]:
            torch.save(model.state_dict(), out / "checkpoints" / f"epoch_{epoch:03d}.pt")
        vrow = run_eval(epoch)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch, "global_step": global_step},
                   out / "checkpoints" / "last.pt")
        card["progress"].append({"epoch": epoch, "train_minutes": round(train_minutes, 1), **vrow})
        card["status"] = "finished" if epoch == tr["epochs"] else f"epoch {epoch} of {tr['epochs']} done"
        yaml.safe_dump(card, open(card_path, "w"), sort_keys=False)
    log("done")


if __name__ == "__main__":
    main()
