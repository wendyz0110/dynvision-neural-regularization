"""
ImageNet-val accuracy of the pretrained CordsNet-R8 and of fine-tuned models (epoch 10).

Preprocessing as in CordsNet's own ImageNet evaluation: Resize(256) + CenterCrop(224),
ImageNet normalization (the paper's Table 1 reports 57.90% top-1 for R8 on ImageNet val).
Images are read with torchvision ImageFolder from the 1000 synset folders of /imagenet/val
(imagenet-val.sqf mounted into the container, see slurm/eval_imagenet.sbatch); folder order
is checked against CordsNet's class order (prepare_images/imagenet_wnids.json).

The model sees each image for 100 stimulus steps after 100 idle steps (as in training);
the top-5 classes are stored at every stimulus step, so accuracy over time can be plotted.

Output, one file per model: <out_dir>/<name>.npz
  top5      (N, T, 5) int16   top-5 ImageNet classes per image and stimulus step
  label     (N,) int16        true class of each image
  top1_t, top5_t   (T,)       accuracy at every step
  time_ms   (T,)              end of each 2 ms step after image onset
  checkpoint                  weights used ("pretrained" = cordsnetr8.pth)
A model whose npz already exists is skipped (rerun the job to finish after a timeout).

Usage (on Torch, GPU job, see slurm/eval_imagenet.sbatch):
  python eval_imagenet.py                                      # the four Round 3 models
  python eval_imagenet.py --models pretrained --max_batches 5  # quick check
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

ROUND3_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND3_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR))
sys.path.insert(0, str(CORDSNET_DIR / "Round1"))

from common.cordsnet_model import build_cordsnet  # noqa: E402
from common.manifest_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from core.recording import N_BINS, BinRecorder  # noqa: E402

# name -> run directory (epoch-10 checkpoint); "pretrained" = the original checkpoint
MODELS = {
    "pretrained": None,
    "control": "CordsNet/Round1/outputs/round1/control",
    "multi3_w1each": "CordsNet/Round2/outputs/multi3_w1each",
    "multi3_w0p33each": "CordsNet/Round2/outputs/multi3_w0p33each",
}
# model settings come from this run's config (all runs are checked to be identical)
REFERENCE_RUN = "CordsNet/Round1/outputs/round1/control"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagenet_val", default="/imagenet/val")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--out_dir", default="CordsNet/Round3/outputs/imagenet_val")
    ap.add_argument("--wnids", default="CordsNet/prepare_images/imagenet_wnids.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=None, help="only for quick checks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    device = torch.device(args.device)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model settings: identical for every run ----
    cfg_model = yaml.safe_load(open(resolve(REFERENCE_RUN) / "config_used.yaml"))["model"]
    for name in args.models:
        if MODELS[name] is not None:
            c = yaml.safe_load(open(resolve(MODELS[name]) / "config_used.yaml"))["model"]
            assert c == cfg_model, f"{name}: model settings differ from {REFERENCE_RUN}"

    # ---- data ----
    tf = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = ImageFolder(args.imagenet_val, transform=tf)
    wnids = [d["wnid"] for d in json.load(open(resolve(args.wnids)))]
    assert ds.classes == wnids, "ImageNet folder order differs from CordsNet's class order"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    n_batches = len(loader) if args.max_batches is None else min(args.max_batches, len(loader))
    n_img = min(len(ds), n_batches * args.batch_size)
    labels = np.array(ds.targets[:n_img], dtype=np.int16)
    log(f"{len(ds)} images in {args.imagenet_val}; evaluating {n_img} ({n_batches} batches)")

    # ---- model (built once; weights swapped per run) ----
    ckpt_path = resolve(cfg_model["checkpoint"])
    mk = {k: v for k, v in cfg_model.items() if k not in ("checkpoint", "n_classes")}
    model, info = build_cordsnet(ckpt_path, trainable=False, n_classes=cfg_model["n_classes"], **mk)
    model = model.to(device).eval()
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    n_t, dt = info["model_kwargs"]["n_timesteps"], info["model_kwargs"]["dt"]
    # Without this wrapper DynVision copies every layer's activity at every step to CPU memory.
    rec = BinRecorder(model, [], steps_per_bin=max(1, n_t // N_BINS))

    for name in args.models:
        path = out_dir / f"{name}.npz"
        if path.exists():
            log(f"{name}: {path} exists, skipped")
            continue
        if MODELS[name] is None:
            model.load_state_dict(pretrained_state)
            ckpt = str(ckpt_path)
        else:
            ckpt = resolve(MODELS[name]) / "checkpoints" / f"epoch_{args.epoch:03d}.pt"
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            ckpt = str(ckpt)
        model.eval()

        t0 = time.time()
        top5 = np.zeros((n_img, n_t, 5), dtype=np.int16)
        start = 0
        with torch.no_grad():
            for b, (x, _) in enumerate(loader):
                if b == n_batches:
                    break
                rec.reset()
                x = x.to(device, non_blocking=True)
                logits = model(x.unsqueeze(1).expand(-1, n_t, -1, -1, -1))              # (B, T, 1000)
                top5[start:start + len(x)] = logits.topk(5, dim=-1).indices.cpu().numpy()
                start += len(x)
                if (b + 1) % 100 == 0 or b + 1 == n_batches:
                    log(f"  {name} batch {b + 1}/{n_batches}")
        assert start == n_img

        hit1 = top5[:, :, 0] == labels[:, None]                                         # (N, T)
        hit5 = (top5 == labels[:, None, None]).any(-1)
        np.savez(path, top5=top5, label=labels, top1_t=hit1.mean(0), top5_t=hit5.mean(0),
                 time_ms=(np.arange(n_t) + 1) * dt, checkpoint=ckpt, epoch=args.epoch,
                 preprocessing="Resize(256) + CenterCrop(224), ImageNet normalization")
        log(f"{name}: last-step top-1 {hit1[:, -1].mean():.4f}, top-5 {hit5[:, -1].mean():.4f} "
            f"({time.time() - t0:.0f}s) -> {path}")
    rec.detach()


if __name__ == "__main__":
    main()
