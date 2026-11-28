"""
Round 4 memory / speed test (as Round3/train/memtest.py, with the Round 4 model: THINGS head over
all 1854 concepts, CE on every THINGS image; batches from core4/batches.py).
On one GPU: one full training step per setting, as in train.py:
  ImageNet part  batch of the same size, forward (no layer recording) + CE on the ImageNet head + backward
  THINGS part    one-day batch from the cache, forward with layer recording, CE on the THINGS head
                 + CKA loss of every target with gradient (worst case for memory) + backward
  then one optimizer step (lr 1e-12, so the weights barely move).
The ImageNet images are random tensors of the right shape (memory and compute depend only on the
shape), so the test does not need ImageNet mounted. fp32 only (the training precision).

Reported per batch size: peak memory of the ImageNet part, of the THINGS part, and of the whole
step (reset between parts: the two graphs are never in memory together), and seconds per step
(mean over the steps after the first).

Usage (on Torch, see ../slurm/memtest.sbatch):
  python memtest.py --config ../configs/multi3_w1each.yaml --batch_sizes 96 --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROUND4_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND4_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
for p in (CORDSNET_DIR, CORDSNET_DIR / "Round1", ROUND4_DIR):
    sys.path.insert(0, str(p))

from common.cordsnet_things import build_cordsnet_things  # noqa: E402
from core.data import Cache  # noqa: E402
from core.losses import cka_loss, cka_s_bins, parse_targets, weighted_ce  # noqa: E402
from core.recording import BinRecorder  # noqa: E402
from core4.batches import train_batches  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[96])
    ap.add_argument("--steps", type=int, default=3, help="steps per setting (first one is warm-up)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lo, mc = cfg["loss"], cfg["model"]
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    targets = parse_targets(lo)
    rec_layers = list(dict.fromkeys(t["layer"] for t in targets))
    cache = Cache(resolve(cfg["data"]["cache_dir"]), device, areas=list(dict.fromkeys(t["area"] for t in targets)))
    mk = {k: v for k, v in mc.items() if k not in ("checkpoint", "things_head")}
    model, heads, info = build_cordsnet_things(resolve(mc["checkpoint"]), resolve(mc["things_head"]), trainable=True, **mk)
    model = model.to(device).train()
    n_t = info["model_kwargs"]["n_timesteps"]
    things_lab = heads.labels(cache.concept, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-12)
    rec = BinRecorder(model, rec_layers, grad_bins={name: sorted({k for t in targets if t["layer"] == name for k in t["bins"]})
                                                    for name in rec_layers})
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else None
    results = {"gpu": gpu, "total_memory_GB": total, "config": args.config, "settings": []}
    print(f"GPU: {gpu} ({total} GB); targets: " + ", ".join(t["tag"] for t in targets), flush=True)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def peak_gb():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None

    for B in args.batch_sizes:
        if cuda:
            torch.cuda.empty_cache()
        res = {"batch_size": B}
        try:
            times, t_im, t_th, peak_im, peak_th = [], [], [], 0.0, 0.0
            for s, (day, rows) in enumerate(train_batches(cache, "train", B, 0, 1)[:args.steps]):
                opt.zero_grad(set_to_none=True)
                sync()
                t0 = time.time()
                # ImageNet part
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                x = torch.randn(B, 3, 224, 224, device=device)
                y = torch.randint(0, 1000, (B,), device=device)
                rec.enabled = False
                rec.reset()
                loss = weighted_ce(model(x.unsqueeze(1).expand(-1, n_t, -1, -1, -1))[..., heads.imagenet], y, *lo["ce_steps"])
                loss.backward()
                rec.enabled = True
                del x, loss
                sync()
                t1 = time.time()
                peak_im = max(peak_im, peak_gb() or 0.0)
                # THINGS part
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                rows_t = torch.as_tensor(rows, device=device)
                rec.reset()
                logits = model(cache.images_for(rows_t).unsqueeze(1).expand(-1, n_t, -1, -1, -1))
                loss = weighted_ce(logits[..., heads.things], things_lab[rows_t], *lo["ce_steps"])
                for tg in targets:                           # every target with gradient (worst case)
                    vals, status = cka_s_bins(rec.bins[tg["layer"]], cache.neural[tg["area"]][rows_t])
                    lc = cka_loss(vals, tg["bins"], lo["floor"])
                    if lc is not None:
                        loss = loss + max(tg["weight"], 1.0) * lc
                loss.backward()
                opt.step()
                del logits, loss
                sync()
                t2 = time.time()
                peak_th = max(peak_th, peak_gb() or 0.0)
                if s > 0:
                    times.append(t2 - t0)
                    t_im.append(t1 - t0)
                    t_th.append(t2 - t1)
            mean = lambda v: round(sum(v) / len(v), 2) if v else None  # noqa: E731
            res.update(ok=True, peak_imagenet_GB=peak_im, peak_things_GB=peak_th, peak_GB=max(peak_im, peak_th),
                       sec_per_step=mean(times), sec_imagenet=mean(t_im), sec_things=mean(t_th))
        except torch.cuda.OutOfMemoryError:
            res.update(ok=False)
        rec.enabled = True
        rec.reset()
        opt.zero_grad(set_to_none=True)
        results["settings"].append(res)
        print(f"  B={B:3d}: " + (f"peak {res['peak_GB']:6.1f} GB (ImageNet part {res['peak_imagenet_GB']:.1f}, "
                                 f"THINGS part {res['peak_things_GB']:.1f}); {res['sec_per_step']} s/step "
                                 f"({res['sec_imagenet']} + {res['sec_things']})" if res["ok"] and cuda
                                 else ("ok (cpu)" if res["ok"] else "out of memory")), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
