"""
Classification accuracy of the Round 1 models, from their saved checkpoints.

The per-epoch evaluation of train.py stored only the CE per step, not the predictions,
so this script reruns every checkpoint (epoch 0 = original pretrained model, epochs 1-10 =
checkpoints/epoch_XXX.pt) on the images of one split (default: val, all 1,854 images) and
stores the top-5 ImageNet classes at every stimulus step. Accuracies are computed in
Round1/analysis/plot_round1.ipynb:
  - agreement with the pseudo-labels (= the pretrained model's last-step prediction)
  - THINGS true-label accuracy on the concepts that overlap ImageNet
    (prepare_images/things_imagenet_overlap.csv; correct if the prediction is one of the
    concept's matched ImageNet classes)

Output, per run: <run_dir>/accuracy.npz
  epochs        (E,)              0, 1, ..., 10
  top5          (E, N, T, 5) int16  top-5 ImageNet classes per epoch, image and stimulus step
  stim_idx, concept, pseudo_label   (N,) for the evaluated images
  overlap_concepts, overlap_idx     concept name and ';'-joined matched ImageNet classes
  time_ms       (T,)              end of each 2 ms step after image onset

Usage (on Torch, GPU job, see slurm/eval_accuracy.sbatch):
  python eval_accuracy.py                                   # all three Round 1 runs
  python eval_accuracy.py --runs control --epochs 0 10      # subset (quick check)
"""
import argparse
import csv
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
from core.data import Cache  # noqa: E402
from core.recording import N_BINS, BinRecorder  # noqa: E402

RUNS = ["control", "student_IT_layer8_w0p1", "student_IT_layer8_w1"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="CordsNet/Round1/outputs/round1")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--epochs", nargs="+", type=int, default=None, help="default: 0 to the last checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--overlap_csv", default="CordsNet/prepare_images/things_imagenet_overlap.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    device = torch.device(args.device)
    runs_dir = resolve(args.runs_dir)

    # all runs share the cache and model settings (checked below)
    cfgs = {run: yaml.safe_load(open(runs_dir / run / "config_used.yaml")) for run in args.runs}
    cfg = cfgs[args.runs[0]]
    for run, c in cfgs.items():
        assert c["data"] == cfg["data"] and c["model"] == cfg["model"], f"{run}: different data or model settings"

    cache = Cache(resolve(cfg["data"]["cache_dir"]), device, areas=())
    rows = np.where(cache.split == args.split)[0]
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    log(f"{len(rows)} {args.split} images, {len(batches)} batches")

    overlap = list(csv.DictReader(open(resolve(args.overlap_csv))))

    ckpt_path = resolve(cfg["model"]["checkpoint"])
    mk = {k: v for k, v in cfg["model"].items() if k not in ("checkpoint", "n_classes")}
    model, info = build_cordsnet(ckpt_path, trainable=False, n_classes=cfg["model"]["n_classes"], **mk)
    model = model.to(device).eval()
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    n_t = info["model_kwargs"]["n_timesteps"]
    dt = info["model_kwargs"]["dt"]
    # Without this wrapper, DynVision in eval mode copies every layer's activity at every
    # step to CPU memory (tens of GB per batch). The recorder (no layers) drops them, as
    # in training and evaluation, and runs the idle steps under no_grad.
    rec = BinRecorder(model, [], steps_per_bin=max(1, n_t // N_BINS))

    for run in args.runs:
        out_dir = runs_dir / run
        n_ckpt = len(sorted((out_dir / "checkpoints").glob("epoch_*.pt")))
        epochs = args.epochs if args.epochs is not None else list(range(n_ckpt + 1))
        top5 = np.zeros((len(epochs), len(rows), n_t, 5), dtype=np.int16)
        for ei, epoch in enumerate(epochs):
            t0 = time.time()
            if epoch == 0:
                model.load_state_dict(pretrained_state)
            else:
                model.load_state_dict(torch.load(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                                                 map_location=device, weights_only=True))
            model.eval()
            start = 0
            with torch.no_grad():
                for b in batches:
                    rec.reset()
                    x = cache.images_for(torch.as_tensor(b, device=device))
                    logits = model(x.unsqueeze(1).expand(-1, n_t, -1, -1, -1))           # (B, T, 1000)
                    top5[ei, start:start + len(b)] = logits.topk(5, dim=-1).indices.cpu().numpy()
                    start += len(b)
            agree = (top5[ei, :, -1, 0] == cache.pseudo_label[torch.as_tensor(rows, device=device)].cpu().numpy()).mean()
            log(f"{run} epoch {epoch}: last-step agreement with pseudo-labels {agree:.4f} ({time.time() - t0:.0f}s)")

        np.savez(out_dir / "accuracy.npz",
                 epochs=np.array(epochs), top5=top5, split=args.split,
                 stim_idx=cache.stim_idx[rows], concept=cache.concept[rows],
                 pseudo_label=cache.pseudo_label.cpu().numpy()[rows],
                 overlap_concepts=np.array([r["uniqueID"] for r in overlap]),
                 overlap_idx=np.array([r["imagenet_idx"] for r in overlap]),
                 time_ms=(np.arange(n_t) + 1) * dt)
        log(f"saved {out_dir / 'accuracy.npz'}")


if __name__ == "__main__":
    main()
