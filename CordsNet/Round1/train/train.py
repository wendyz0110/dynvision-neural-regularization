"""
Fine-tune the pretrained CordsNet-R8 with
    loss = w_ce * CE(pseudo-labels) + sum over alignment targets of
           weight * mean_k log(1 - CKA_S(layer, area, bin k))
Round 1: one target (IT vs layer8); Round 2: several (e.g. V1-layer2, V4-layer6, IT-layer8),
`loss.targets` in the config (core/losses.py: parse_targets; Round 1 configs still work).
Student: weight > 0. Control: weight 0 (same script, same seed -> same batches; CKA is still
computed and logged, without gradient).

Every epoch: checkpoint, evaluation on the val split with the epoch-0 protocol (all layers x
areas x bins, three CKA estimators, CE per step), model card update. Epoch 0 (before any
update) is evaluated too, which should reproduce Round1/outputs/epoch0_pretrained.
Runs resume from <output_dir>/checkpoints/last.pt automatically.

Outputs (output_dir):
  model_card.yaml            full configuration, inputs with sha256, environment, progress
  train_metrics.csv          one row per training step
  val_metrics.csv            one row per evaluation (epoch 0, 1, ...): CE, CKA_S per target
                             (pooled over the target's bins), accuracy at 200 ms (agreement
                             with the pseudo-labels, THINGS top-1 / top-5)
  eval/epoch_XXX.npz         full evaluation arrays (same keys as epoch-0 results.npz)
  checkpoints/epoch_XXX.pt   model weights after each epoch
  checkpoints/last.pt        model + optimizer + scheduler + RNG state, for resuming

Usage:
  python train.py --config configs/student_IT_layer8_w0p1.yaml
  python train.py --config configs/control.yaml
  # short pilot runs override fields on the command line, e.g.
  python train.py --config configs/control.yaml --lr 3e-5 --epochs 1 --max_steps 200 \
      --output_dir ../outputs/round1/pilot_ce_lr3e-5
"""
import argparse
import csv
import hashlib
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROUND1_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND1_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR))
sys.path.insert(0, str(ROUND1_DIR))

from common.cordsnet_model import build_cordsnet  # noqa: E402
from core.accuracy import accuracy, load_overlap  # noqa: E402
from core.data import Cache, train_batches  # noqa: E402
from core.evaluate import evaluate  # noqa: E402
from core.losses import as_float, cka_loss, cka_s_bins, parse_targets, weighted_ce  # noqa: E402
from core.recording import N_BINS, BinRecorder  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"unknown ({e})"


def window_mean(arr, layer_i, area_j, bins, est):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(arr[layer_i, area_j, bins, est]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--w_cka", type=float, default=None, help="weight of every alignment target")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="stop each epoch after N steps (pilots)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tr, lo, ev = cfg["train"], cfg["loss"], cfg["eval"]
    for key, sect in [("lr", tr), ("epochs", tr), ("batch_size", tr), ("precision", tr),
                      ("max_steps", tr), ("device", cfg["runtime"])]:
        if getattr(args, key) is not None:
            sect[key] = getattr(args, key)
    if args.w_cka is not None:
        for t in lo.get("targets", [lo]):
            t["weight" if "targets" in lo else "w_cka"] = args.w_cka
    out = Path(args.output_dir).resolve() if args.output_dir else resolve(cfg["runtime"]["output_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    device = torch.device(cfg["runtime"]["device"])
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731

    # ---- data ----
    t0 = time.time()
    cache = Cache(resolve(cfg["data"]["cache_dir"]), device, areas=ev["areas"])
    log(f"cache on {device} in {time.time() - t0:.0f}s: {len(cache.stim_idx)} images")

    # ---- model ----
    ckpt_path = resolve(cfg["model"]["checkpoint"])
    mk = {k: v for k, v in cfg["model"].items() if k not in ("checkpoint", "n_classes")}
    model, model_info = build_cordsnet(ckpt_path, trainable=True, n_classes=cfg["model"]["n_classes"], **mk)
    model = model.to(device).train()
    n_t = model_info["model_kwargs"]["n_timesteps"]
    assert n_t % N_BINS == 0, "stimulus steps must be a multiple of the 20 bins"
    spb, ce_steps = n_t // N_BINS, tuple(cfg["loss"]["ce_steps"])
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"model: {sum(p.numel() for p in params):,} trainable parameters")

    B, seed = tr["batch_size"], tr["seed"]
    steps_per_epoch = len(train_batches(cache, "train", B, seed, 1))
    if tr.get("max_steps"):
        steps_per_epoch = min(steps_per_epoch, tr["max_steps"])
    total_steps = steps_per_epoch * tr["epochs"]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=(tr["precision"] == "bf16"))

    targets = parse_targets(lo)
    for t in targets:
        assert t["area"] in ev["areas"] and t["layer"] in ev["layers"], f"{t['tag']} not in eval areas / layers"
        t["li"], t["aj"] = ev["layers"].index(t["layer"]), ev["areas"].index(t["area"])
    single = len(targets) == 1                       # Round 1 form: also write the Round 1 column names
    student = any(t["weight"] > 0 for t in targets)
    rec_layers = list(dict.fromkeys(t["layer"] for t in targets))
    grad_bins = {name: sorted({k for t in targets if t["layer"] == name and t["weight"] > 0 for k in t["bins"]})
                 for name in rec_layers}
    overlap_path = resolve(ev.get("overlap_csv", "CordsNet/prepare_images/things_imagenet_overlap.csv"))
    overlap = load_overlap(overlap_path) if overlap_path.exists() else None
    log("targets: " + ", ".join(f"{t['tag']} bins {t['bins'][0]}-{t['bins'][-1]} weight {t['weight']}" for t in targets))
    log(f"THINGS accuracy: {overlap_path if overlap else 'overlap CSV not found, skipped'}")

    # ---- resume ----
    start_epoch, global_step = 1, 0
    last = out / "checkpoints" / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["torch_rng"])
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
        "description": ("student: CE + CKA_S alignment (" + ", ".join(f"{t['tag']} x {t['weight']}" for t in targets) + ")"
                        if student else "control: CE only (CKA logged, no gradient)"),
        "init_checkpoint": {"path": str(ckpt_path), "sha256": sha256(ckpt_path)},
        "model": model_info,
        "data": {"cache_dir": str(resolve(cfg["data"]["cache_dir"])),
                 "cache_info": yaml.safe_load(open(resolve(cfg["data"]["cache_dir"]) / "cache_info.json")),
                 "train_split": "train", "train_images": int((cache.split == "train").sum()),
                 "batches": "one recording day per batch; per-epoch shuffle with seed (seed, epoch); partial batches dropped",
                 "augmentation": "none (CKA needs the exact images the monkey saw; pseudo-labels come from unaugmented images)"},
        "loss": {**lo, "targets_parsed": [{k: t[k] for k in ("area", "layer", "weight", "tag")} | {"bins": [t["bins"][0], t["bins"][-1]]}
                                          for t in targets],
                 "ce": "weighted mean over outputs 70-99 (142-200 ms), weights logspace(-3,0,30)/sum, pseudo-labels",
                 "cka": "mean over valid bins of log(max(1 - CKA_S, floor)); CKA_S = Song et al. 2012; "
                        "bins skipped if undefined (no variance across images) or failed (Song denominator <= 0)",
                 "activations": "DynVision `record` of the layer (post-ReLU firing rate, before pooling), mean over the 5 steps of each bin"},
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
    val_cols = (["epoch", "global_step", "val_ce"] + (["CKA_0", "CKA_S", "CKA_C"] if single else [])
                + [f"CKA_S_{t['tag']}" for t in targets] + ["agree", "things_top1", "things_top5", "minutes"])
    train_cols = (["epoch", "step", "global_step", "day", "lr", "loss_total", "loss_ce"]
                  + (["loss_cka", "n_undefined", "n_failed"] if single else [])
                  + ["grad_norm", "clipped", "seconds"]
                  + ([f"cka_S_bin{k:02d}" for k in range(N_BINS)] if single else [])
                  + [c for t in targets for c in [f"loss_cka_{t['tag']}", f"n_undefined_{t['tag']}", f"n_failed_{t['tag']}"]
                                                 + [f"cka_S_{t['tag']}_bin{k:02d}" for k in range(N_BINS)]])

    def run_eval(epoch):
        t = time.time()
        log(f"evaluating epoch {epoch} on {ev['splits']}")
        r = evaluate(model, cache, ev["layers"], ev["areas"], tuple(ev["splits"]), ev["batch_size"],
                     ev["seed"], n_t, ce_steps, log=log)
        np.savez(out / "eval" / f"epoch_{epoch:03d}.npz", **r)
        row = {"epoch": epoch, "global_step": global_step, "val_ce": round(r["ce_weighted"], 5),
               "minutes": round((time.time() - t) / 60, 2)}
        if single:
            for e, est in enumerate(["CKA_0", "CKA_S", "CKA_C"]):
                row[est] = round(window_mean(r["cka_pooled"], targets[0]["li"], targets[0]["aj"], targets[0]["bins"], e), 5)
        for tg in targets:
            row[f"CKA_S_{tg['tag']}"] = round(window_mean(r["cka_pooled"], tg["li"], tg["aj"], tg["bins"], 1), 5)
        acc = accuracy(r["top5"], r["eval_rows"], cache.pseudo_label.cpu().numpy(), cache.concept, overlap)
        row.update({k: round(acc[k], 5) for k in ("agree", "things_top1", "things_top5")})
        new = not val_path.exists()
        with open(val_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=val_cols)
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"epoch {epoch}: val CE {row['val_ce']:.4f}; CKA_S (pooled over each target's bins): "
            + "  ".join(f"{tg['tag']} {row['CKA_S_' + tg['tag']]:.4f}" for tg in targets)
            + f"; agreement {row['agree']:.4f}, THINGS top-1 {row['things_top1']:.4f} (n = {acc['things_n']})")
        return row

    if start_epoch == 1 and ev.get("at_start", True) and not (out / "eval" / "epoch_000.npz").exists():
        row = run_eval(0)
        card["progress"].append({"epoch": 0, **row})
        yaml.safe_dump(card, open(card_path, "w"), sort_keys=False)

    # ---- train ----
    for epoch in range(start_epoch, tr["epochs"] + 1):
        batches = train_batches(cache, "train", B, seed, epoch)[:steps_per_epoch]
        # only each target's loss bins stay in the graph (none for weight 0); other bins are logged only
        rec = BinRecorder(model, rec_layers, grad_bins=grad_bins, steps_per_bin=spb)
        te = time.time()
        rows_out = []
        for step, (day, rows) in enumerate(batches, start=1):
            ts = time.time()
            rows_t = torch.as_tensor(rows, device=device)
            rec.reset()
            x = cache.images_for(rows_t)
            with torch.autocast(**amp):
                logits = model(x.unsqueeze(1).expand(-1, n_t, -1, -1, -1))
            rec.check_complete(n_t)
            loss_ce = weighted_ce(logits, cache.pseudo_label[rows_t], *ce_steps)
            loss = lo["w_ce"] * loss_ce
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"]))
            lr_used = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            global_step += 1

            row = {"epoch": epoch, "step": step, "global_step": global_step, "day": day,
                   "lr": lr_used, "loss_total": loss.item(), "loss_ce": loss_ce.item(),
                   "grad_norm": gn, "clipped": int(gn > tr["grad_clip"]), "seconds": round(time.time() - ts, 2)}
            for tg in targets:
                vals, status, loss_cka = res[tg["tag"]]
                in_loss = [status[k] for k in tg["bins"]]
                row.update({f"loss_cka_{tg['tag']}": as_float(loss_cka),
                            f"n_undefined_{tg['tag']}": in_loss.count("undefined"),
                            f"n_failed_{tg['tag']}": in_loss.count("failed")})
                row.update({f"cka_S_{tg['tag']}_bin{k:02d}": as_float(v) for k, v in enumerate(vals)})
                if single:
                    row.update({"loss_cka": as_float(loss_cka), "n_undefined": in_loss.count("undefined"),
                                "n_failed": in_loss.count("failed")})
                    row.update({f"cka_S_bin{k:02d}": as_float(v) for k, v in enumerate(vals)})
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
                log(f"epoch {epoch} step {step}/{len(batches)}  loss {row['loss_total']:.4f}  CE {row['loss_ce']:.4f}  "
                    + "  ".join(f"L_{tg['tag']} {row['loss_cka_' + tg['tag']]:.4f}" for tg in targets)
                    + f"  grad {gn:.1f}  {row['seconds']:.1f}s/step  "
                    f"~{el / step * (len(batches) - step) / 60:.0f} min left in epoch")
        rec.detach()
        train_minutes = (time.time() - te) / 60

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
