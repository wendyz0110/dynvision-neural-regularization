"""
Cross-monkey evaluation: Round 1 checkpoints (trained on monkey N) evaluated against another
monkey's neural data (default: monkey F).

Same protocol and code as the per-epoch evaluation during training (core/evaluate.py): val
split, batches of 64 from one recording day of the evaluated monkey (here F's days), all
layers x areas x 20 bins, pooled CKA. Only the neural data change: the cache of the other
monkey (same images, same train/val/test split as monkey N).

Epoch 0 = the original pretrained model, epochs 1-10 = checkpoints/epoch_XXX.pt.
Epoch 0 must equal Round1/outputs/epoch0_pretrained_monkeyF (same batches); checked in the
plotting notebook.

Output, per run and epoch: <run_dir>/eval_<monkey>/epoch_XXX.npz (same keys as
<run_dir>/eval/epoch_XXX.npz). Finished epochs are skipped, so the job can be resubmitted.

Usage (on Torch, GPU job, see slurm/eval_other_monkey.sbatch):
  python eval_other_monkey.py                                       # 3 runs x epochs 0-10 on monkey F
  python eval_other_monkey.py --runs control --epochs 0 --max_batches 2   # quick check
"""
import argparse
import sys
import time
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
import core.evaluate as core_eval  # noqa: E402
from core.data import Cache, eval_batches  # noqa: E402
from core.evaluate import evaluate  # noqa: E402

RUNS = ["control", "student_IT_layer8_w0p1", "student_IT_layer8_w1"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="CordsNet/Round1/outputs/round1")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--epochs", nargs="+", type=int, default=None, help="default: 0 to the last checkpoint")
    ap.add_argument("--monkey", default="monkeyF")
    ap.add_argument("--cache_dir", default="/scratch/wz2723/data/TVSD/round1_cache_monkeyF")
    ap.add_argument("--max_batches", type=int, default=None, help="quick check only (output not saved)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    device = torch.device(args.device)
    runs_dir = resolve(args.runs_dir)

    cfgs = {run: yaml.safe_load(open(runs_dir / run / "config_used.yaml")) for run in args.runs}
    cfg = cfgs[args.runs[0]]
    for run, c in cfgs.items():
        assert c["model"] == cfg["model"] and c["eval"] == cfg["eval"], f"{run}: different model or eval settings"
    ev = cfg["eval"]

    cache = Cache(resolve(args.cache_dir), device, areas=tuple(ev["areas"]))
    log(f"{args.monkey} cache {args.cache_dir}: " + ", ".join(f"{a} {tuple(cache.neural[a].shape)}" for a in ev["areas"]))
    if args.max_batches:                                        # quick check: first batches only
        core_eval.eval_batches = lambda *a, **k: eval_batches(*a, **k)[:args.max_batches]

    mk = {k: v for k, v in cfg["model"].items() if k not in ("checkpoint", "n_classes")}
    model, info = build_cordsnet(resolve(cfg["model"]["checkpoint"]), trainable=False,
                                 n_classes=cfg["model"]["n_classes"], **mk)
    model = model.to(device).eval()
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    n_t = info["model_kwargs"]["n_timesteps"]
    ce_steps = tuple(cfg["loss"]["ce_steps"])

    for run in args.runs:
        run_dir = runs_dir / run
        out_dir = run_dir / f"eval_{args.monkey}"
        out_dir.mkdir(exist_ok=True)
        n_ckpt = len(sorted((run_dir / "checkpoints").glob("epoch_*.pt")))
        epochs = args.epochs if args.epochs is not None else list(range(n_ckpt + 1))
        for epoch in epochs:
            path = out_dir / f"epoch_{epoch:03d}.npz"
            if path.exists() and not args.max_batches:
                log(f"{run} epoch {epoch}: done already, skipped")
                continue
            t0 = time.time()
            if epoch == 0:
                model.load_state_dict(pretrained_state)
            else:
                model.load_state_dict(torch.load(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                                                 map_location=device, weights_only=True))
            r = evaluate(model, cache, ev["layers"], ev["areas"], tuple(ev["splits"]), ev["batch_size"],
                         ev["seed"], n_t, ce_steps, log=log)
            li, aj = list(r["layers"]).index("layer8"), list(r["areas"]).index("IT")
            it8 = np.nanmean(r["cka_pooled"][li, aj, 7:, 1])
            log(f"{run} epoch {epoch}: {len(r['batch_day'])} batches, IT-layer8 CKA_S (70-200 ms) {it8:.4f} "
                f"({(time.time() - t0) / 60:.1f} min)")
            if not args.max_batches:
                np.savez(path, **r, monkey=args.monkey, cache_dir=str(args.cache_dir))


if __name__ == "__main__":
    main()
