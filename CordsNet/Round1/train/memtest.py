"""
Memory and speed test for Round 1 training: for each batch size and precision, run a few
full training steps (forward over 100 idle + 100 stimulus steps, CE + CKA_S loss on every
alignment target of the config, backward, optimizer step) on real cached data and record the peak
GPU memory and the time per step. Also checks how much bf16 changes the model output
(argmax agreement with fp32 on one batch).

Usage (on a GPU node):
  python memtest.py --config configs/student_IT_layer8_w0p1.yaml --batch_sizes 16 32 48 64 \
      --precisions bf16 fp32 --out ../outputs/round1/memtest_<gpu>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROUND1_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND1_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR))
sys.path.insert(0, str(ROUND1_DIR))

from common.cordsnet_model import build_cordsnet  # noqa: E402
from core.data import Cache, train_batches  # noqa: E402
from core.losses import cka_loss, cka_s_bins, parse_targets, weighted_ce  # noqa: E402
from core.recording import BinRecorder  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--precisions", nargs="+", default=["bf16", "fp32"])
    ap.add_argument("--steps", type=int, default=3, help="steps per setting (first one is warm-up)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lo, tr = cfg["loss"], cfg["train"]
    device = torch.device("cuda")
    targets = parse_targets(lo)
    rec_layers = list(dict.fromkeys(t["layer"] for t in targets))
    cache = Cache(resolve(cfg["data"]["cache_dir"]), device, areas=list(dict.fromkeys(t["area"] for t in targets)))
    mk = {k: v for k, v in cfg["model"].items() if k not in ("checkpoint", "n_classes")}
    model, info = build_cordsnet(resolve(cfg["model"]["checkpoint"]), trainable=True,
                                 n_classes=cfg["model"]["n_classes"], **mk)
    model = model.to(device).train()
    n_t = info["model_kwargs"]["n_timesteps"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-12)          # tiny lr: the weights barely move during the test
    rec = BinRecorder(model, rec_layers, grad_bins={name: sorted({k for t in targets if t["layer"] == name for k in t["bins"]})
                                                    for name in rec_layers})
    gpu = torch.cuda.get_device_name(0)
    results = {"gpu": gpu, "total_memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
               "settings": []}
    print(f"GPU: {gpu} ({results['total_memory_GB']} GB)")

    # bf16 vs fp32 output on one batch (no grad)
    day, rows = train_batches(cache, "train", 32, 0, 1)[0]
    rows_t = torch.as_tensor(rows, device=device)
    x = cache.images_for(rows_t).unsqueeze(1).expand(-1, n_t, -1, -1, -1)
    with torch.no_grad():
        rec.reset()
        out32 = model(x)
        rec.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out16 = model(x)
    agree = float((out32[:, -1].argmax(-1) == out16[:, -1].float().argmax(-1)).float().mean())
    pl = float((out32[:, -1].argmax(-1) == cache.pseudo_label[rows_t]).float().mean())
    results["bf16_vs_fp32_last_step_argmax_agreement"] = agree
    results["fp32_vs_pseudo_label_agreement"] = pl
    print(f"last-step argmax agreement: bf16 vs fp32 {agree:.3f}; fp32 vs pseudo-labels {pl:.3f}")
    del out32, out16, x

    for prec in args.precisions:
        for B in args.batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            res = {"precision": prec, "batch_size": B}
            try:
                times = []
                for s, (day, rows) in enumerate(train_batches(cache, "train", B, 0, 1)[:args.steps]):
                    torch.cuda.synchronize()
                    t = time.time()
                    rows_t = torch.as_tensor(rows, device=device)
                    rec.reset()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(prec == "bf16")):
                        logits = model(cache.images_for(rows_t).unsqueeze(1).expand(-1, n_t, -1, -1, -1))
                    loss = weighted_ce(logits, cache.pseudo_label[rows_t], *lo["ce_steps"])
                    for tg in targets:                    # every target with gradient (worst case for memory)
                        vals, status = cka_s_bins(rec.bins[tg["layer"]], cache.neural[tg["area"]][rows_t])
                        lc = cka_loss(vals, tg["bins"], lo["floor"])
                        if lc is not None:
                            loss = loss + max(tg["weight"], 1.0) * lc
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    times.append(time.time() - t)
                res.update(ok=True, peak_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           sec_per_step=round(sum(times[1:]) / max(len(times) - 1, 1), 2))
            except torch.cuda.OutOfMemoryError:
                res.update(ok=False, peak_GB=None, sec_per_step=None)
            rec.reset()
            opt.zero_grad(set_to_none=True)
            results["settings"].append(res)
            print(f"  {prec:5s} B={B:3d}: " + (f"peak {res['peak_GB']:6.1f} GB, {res['sec_per_step']:6.2f} s/step"
                                               if res["ok"] else "out of memory"), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
