"""
Features for the new THINGS classification head: the input of CordsNet's classifier layer
(layer7 + layer8 -> global average pool -> 512 values) of the frozen pretrained model, at
the CE steps 70-99, for the THINGS concepts that do NOT overlap ImageNet.

The new head runs parallel to the ImageNet head on this same 512-d vector, and the backbone
stays frozen, so the features are fixed: computing them once lets the head be trained on
them directly (train_head.py), without running the model again.

Images: the Round 1 cache (TVSD train images, Resize 224x224, ImageNet normalization; the
same images as the pseudo-labels and CKA). Splits: train (10 images per concept) and val
(1 per concept); test is not touched.

Check: the stored features (first and last step) times the ImageNet head's weights must reproduce the
model's ImageNet logits at those steps (printed; the job stops if they differ).

Output: <out_dir>/features.npz
  features    (N, 30, 512) float32   classifier input at stimulus steps 70-99
  label       (N,) int16             class index 0..C-1 (concepts sorted by name)
  split       (N,)                   train / val
  stim_idx, concept (N,)
  classes     (C,)                   concept names, index = label
  steps       (30,)                  70..99 (stimulus step; output t = (t + 1) * 2 ms)
  imagenet_W (1000, 512), imagenet_b (1000,)   effective weights of the ImageNet head

Usage (on Torch, GPU job, see ../slurm/things_features.sbatch):
  python extract_features.py
  python extract_features.py --max_batches 2 --out_dir CordsNet/Round3/outputs/things_head_check
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROUND3_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND3_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR))
sys.path.insert(0, str(CORDSNET_DIR / "Round1"))

from common.cordsnet_model import build_cordsnet  # noqa: E402
from core.data import Cache  # noqa: E402
from core.recording import N_BINS, BinRecorder  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="/scratch/wz2723/data/TVSD/round1_cache_monkeyN")
    ap.add_argument("--checkpoint", default="CordsNet/checkpoints/cordsnetr8.pth")
    ap.add_argument("--overlap_csv", default="CordsNet/prepare_images/things_imagenet_overlap.csv")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--steps", nargs=2, type=int, default=[70, 100], help="stimulus steps [first, last)")
    ap.add_argument("--out_dir", default="CordsNet/Round3/outputs/things_head")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=None, help="only for quick checks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    device = torch.device(args.device)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s0, s1 = args.steps

    # ---- images of the non-overlapping concepts ----
    cache = Cache(args.cache_dir, device, areas=())
    overlap = {r["uniqueID"] for r in csv.DictReader(open(resolve(args.overlap_csv)))}
    concepts = np.array([str(c) for c in cache.concept])
    classes = sorted(set(concepts) - overlap)
    class_idx = {c: i for i, c in enumerate(classes)}
    rows = np.where(np.isin(cache.split, args.splits) & ~np.isin(concepts, list(overlap)))[0]
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
        rows = np.concatenate(batches)
    log(f"{len(set(concepts))} concepts, {len(overlap)} overlap ImageNet -> {len(classes)} classes; "
        f"{len(rows)} images ({', '.join(f'{s} {int((cache.split[rows] == s).sum())}' for s in args.splits)})")

    # ---- frozen pretrained model ----
    model, info = build_cordsnet(resolve(args.checkpoint), trainable=False)
    model = model.to(device).eval()
    n_t = info["model_kwargs"]["n_timesteps"]
    rec = BinRecorder(model, [], steps_per_bin=max(1, n_t // N_BINS))   # keeps DynVision from storing responses on CPU
    fc = model.classifier[3]                                            # weight-normed Linear(512, 1000)
    assert isinstance(fc, torch.nn.Linear) and fc.in_features == 512, f"unexpected classifier layer: {fc}"
    W, b = fc.weight.detach().clone(), fc.bias.detach().clone()          # effective (weight-norm applied) weights

    # the classifier's input at every stimulus step, keyed by the step (the recorder's counter
    # is the index of the step being computed); idle steps are skipped
    grabbed = {}

    def grab(module, inputs):
        if not model._in_idle_period:
            grabbed[rec.t] = inputs[0].detach()

    hook = fc.register_forward_pre_hook(grab)

    feats = np.zeros((len(rows), s1 - s0, 512), dtype=np.float32)
    max_diff, start, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for k, b_rows in enumerate(batches):
            rec.reset()
            grabbed.clear()
            x = cache.images_for(torch.as_tensor(b_rows, device=device))
            logits = model(x.unsqueeze(1).expand(-1, n_t, -1, -1, -1))           # (B, T, 1000)
            assert sorted(grabbed) == list(range(n_t)), f"classifier inputs at steps {sorted(grabbed)[:5]}..."
            f = torch.stack([grabbed[t] for t in range(s0, s1)], dim=1)            # (B, 30, 512)
            # check: features x ImageNet head = the model's logits (first and last stored step)
            for i, t in [(0, s0), (-1, s1 - 1)]:
                max_diff = max(max_diff, float((f[:, i] @ W.T + b - logits[:, t]).abs().max()))
            feats[start:start + len(b_rows)] = f.cpu().numpy()
            start += len(b_rows)
            if (k + 1) % 20 == 0 or k + 1 == len(batches):
                log(f"  batch {k + 1}/{len(batches)}")
    hook.remove()
    rec.detach()

    log(f"check: max |features x ImageNet head - model logits| = {max_diff:.2e}")
    assert max_diff < 1e-3, "stored features do not reproduce the model's logits"

    path = out_dir / "features.npz"
    np.savez(path, features=feats, label=np.array([class_idx[c] for c in concepts[rows]], dtype=np.int16),
             split=cache.split[rows], stim_idx=cache.stim_idx[rows], concept=concepts[rows],
             classes=np.array(classes), steps=np.arange(s0, s1),
             imagenet_W=W.cpu().numpy(), imagenet_b=b.cpu().numpy(),
             checkpoint=str(resolve(args.checkpoint)), cache_dir=str(args.cache_dir))
    log(f"saved {path} ({feats.nbytes / 1e9:.2f} GB features, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
