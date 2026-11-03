"""
Run a frozen pretrained CordsNet on every image of a stimulus manifest and save
its last-timestep ImageNet logits plus derived quantities.

Outputs (in runtime.output_dir):
  outputs.npz          one row per manifest row (same order):
                         logits      (N, 1000) float32, last stimulus timestep
                         pred_label  (N,)      argmax ImageNet class index (0-999)
                         max_prob    (N,)      largest softmax probability
                         entropy     (N,)      softmax entropy in nats (max ln 1000 = 6.908)
                         entropy_t   (N, T)    softmax entropy at every stimulus timestep;
                                               column t = (t + 1) * dt ms after stimulus onset,
                                               so entropy_t[:, -1] == entropy
                         time_ms     (T,)      time after stimulus onset of each column
                         split, stim_idx, concept   copied from the manifest
  resolved_config.yaml the full configuration actually used (paths resolved, model settings)
  git_commit.txt       commit of the code that produced the outputs
  summary.json         quick statistics

Usage:
  python run_inference.py --config configs/things_cordsnetr8.yaml
  python run_inference.py --config configs/things_cordsnetr8.yaml \
      --limit 4 --device cpu --num_workers 0 --output_dir /tmp/smoke   # smoke test
"""
import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

CORDSNET_DIR = Path(__file__).resolve().parents[1]   # .../CordsNet
REPO_ROOT = CORDSNET_DIR.parent                       # repository root
sys.path.insert(0, str(CORDSNET_DIR))

from common.cordsnet_model import build_cordsnet, repeat_over_time  # noqa: E402
from common.manifest_dataset import ManifestDataset, build_transform  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as e:  # git missing or not a repo
        return f"unknown ({e})"


def imagenet_class_names():
    """ImageNet-1k class names bundled with torchvision (no download needed)."""
    try:
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None, help="override runtime.output_dir")
    ap.add_argument("--device", default=None, help="override runtime.device (cuda / cpu)")
    ap.add_argument("--num_workers", type=int, default=None, help="override runtime.num_workers")
    ap.add_argument("--batch_size", type=int, default=None, help="override runtime.batch_size")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest rows (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing outputs.npz")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg, model_cfg, rt = cfg["data"], dict(cfg["model"]), cfg["runtime"]
    for key in ["output_dir", "device", "num_workers", "batch_size"]:
        if getattr(args, key) is not None:
            rt[key] = getattr(args, key)

    out_dir = Path(rt["output_dir"])
    if (out_dir / "outputs.npz").exists() and not args.overwrite:
        sys.exit(f"{out_dir / 'outputs.npz'} already exists; use --overwrite or another --output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(rt["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("runtime.device is cuda but no GPU is visible (use --device cpu for a CPU test)")

    # ---- model ----
    checkpoint = resolve(model_cfg.pop("checkpoint"))
    n_classes = model_cfg.pop("n_classes")
    t0 = time.time()
    model, model_info = build_cordsnet(
        checkpoint, trainable=False, n_classes=n_classes, image_size=data_cfg["image_size"], **model_cfg
    )
    model = model.to(device)
    n_t = model_info["model_kwargs"]["n_timesteps"]
    print(f"Model ready in {time.time() - t0:.0f}s: verified {model_info['checkpoint_tensors_verified']} "
          f"checkpoint tensors from {checkpoint}")

    # ---- data ----
    dataset = ManifestDataset(
        data_cfg["manifest"], splits=data_cfg.get("splits"),
        transform=build_transform(data_cfg["image_size"]), limit=args.limit,
    )
    loader = DataLoader(
        dataset, batch_size=rt["batch_size"], shuffle=False, num_workers=rt["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    n = len(dataset)
    print(f"{n} images from {data_cfg['manifest']} (splits={data_cfg.get('splits')}), "
          f"{len(loader)} batches of {rt['batch_size']} on {device}")

    # ---- inference ----
    logits_all = np.zeros((n, n_classes), dtype=np.float32)
    entropy_t_all = np.zeros((n, n_t), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    t0 = time.time()
    with torch.inference_mode():
        for b, (images, rows) in enumerate(loader):
            x = repeat_over_time(images.to(device, non_blocking=True), n_t)
            out = model(x)                                  # (B, n_timesteps, n_classes)
            assert out.shape[1] == n_t and out.shape[2] == n_classes, f"unexpected output {tuple(out.shape)}"
            rows = rows.numpy()
            logits_all[rows] = out[:, -1].float().cpu().numpy()
            # entropy at every timestep, computed on the device; only (B, T) numbers are kept
            log_p_t = torch.log_softmax(out.float(), dim=-1)
            entropy_t_all[rows] = (-(log_p_t.exp() * log_p_t).sum(dim=-1)).cpu().numpy()
            seen[rows] = True
            if (b + 1) % rt["log_every"] == 0 or b + 1 == len(loader):
                done = int(seen.sum())
                el = time.time() - t0
                print(f"  batch {b + 1}/{len(loader)}  images {done}/{n}  "
                      f"{el / 60:.1f} min elapsed, ~{el / done * (n - done) / 60:.1f} min left", flush=True)
    assert seen.all(), "some manifest rows were not processed"
    elapsed = time.time() - t0

    # ---- derived quantities ----
    logits_t = torch.from_numpy(logits_all)
    log_p = torch.log_softmax(logits_t, dim=1)
    probs = log_p.exp()
    entropy = (-(probs * log_p).sum(dim=1)).numpy()
    max_prob, pred = probs.max(dim=1)
    max_prob, pred = max_prob.numpy(), pred.numpy()
    assert np.allclose(entropy_t_all[:, -1], entropy, atol=1e-3), "per-timestep entropy disagrees with last-step entropy"
    dt = model_info["model_kwargs"]["dt"]
    time_ms = (np.arange(n_t) + 1) * dt

    # ---- provenance (written before outputs.npz, so outputs.npz existing means the run completed) ----
    resolved = {
        "data": {**data_cfg, "n_images": n, "limit": args.limit},
        "model": model_info,
        "runtime": {**rt, "output_dir": str(out_dir.resolve())},
        "readout": {
            "logits": "last stimulus timestep only",
            "entropy_t": "every stimulus timestep; column t = (t + 1) * dt ms after onset",
            "entropy_units": "nats",
        },
        "versions": {"torch": str(torch.__version__), "python": sys.version.split()[0]},
        "config_file": str(Path(args.config).resolve()),
    }
    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    (out_dir / "git_commit.txt").write_text(git_commit() + "\n")

    names = imagenet_class_names()
    top = Counter(pred.tolist()).most_common(10)
    summary = {
        "n_images": n,
        "inference_minutes": round(elapsed / 60, 2),
        "entropy_nats": {
            "mean": float(entropy.mean()), "median": float(np.median(entropy)),
            "min": float(entropy.min()), "max": float(entropy.max()),
            "uniform_max_ln1000": math.log(n_classes),
        },
        "mean_entropy_by_time_ms": {
            f"{time_ms[i]:g}": float(entropy_t_all[:, i].mean())
            for i in sorted({min(n_t - 1, max(0, int(round(ms / dt)) - 1)) for ms in [20, 50, 100, 140, 150, 200]})
        },
        "max_prob": {"mean": float(max_prob.mean()), "median": float(np.median(max_prob))},
        "n_distinct_predicted_classes": int(len(set(pred.tolist()))),
        "top10_predicted_classes": [
            {"class_idx": c, "name": names[c] if names else None, "count": k} for c, k in top
        ],
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    np.savez(
        out_dir / "outputs.npz",
        logits=logits_all, pred_label=pred, max_prob=max_prob, entropy=entropy,
        entropy_t=entropy_t_all, time_ms=time_ms.astype(np.float32),
        split=np.array(dataset.split), stim_idx=np.array(dataset.stim_idx), concept=np.array(dataset.concept),
    )

    print(f"\nDone in {elapsed / 60:.1f} min. Saved to {out_dir}")
    print(f"entropy (nats): mean {summary['entropy_nats']['mean']:.3f}, median "
          f"{summary['entropy_nats']['median']:.3f}, range [{entropy.min():.3f}, {entropy.max():.3f}] "
          f"(uniform = {math.log(n_classes):.3f})")
    print(f"max prob: mean {max_prob.mean():.3f};  distinct predicted classes: "
          f"{summary['n_distinct_predicted_classes']}")
    print("most frequent predictions:", ", ".join(
        f"{t['name'] or t['class_idx']} ({t['count']})" for t in summary["top10_predicted_classes"][:5]))


if __name__ == "__main__":
    main()
