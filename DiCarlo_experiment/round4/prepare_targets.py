"""
Build the TARGET cache: the frozen target model's outputs for every image in
an image cache (made by prepare_images.py), computed once.

ROUND 4 copy of DiCarlo_experiment/prepare_targets.py (round 2/3, unchanged).
The only difference: the target is given as a trained run folder
(--target_run, made by round4/train_target.py) and rebuilt from its
model_card.yaml with train_target.load_trained_target(), so ANY model class
works (round 4: DyRCNNx8). The round 2 version always built a DyRCNNx4.
The model card is copied into the cache as target_model_card.yaml and its
class is recorded in manifest.json.

What is stored (in --out_dir)
-----------------------------
    logits.npy              [N, T, n_classes] float32, ALL input timesteps.
                            Pseudo-label = argmax(logits[:, -1]) (the target's
                            final prediction); derived at training time.
    <layer>_t<tt>.npy       [N, C, H, W], one file per (layer, timestep),
                            e.g. IT_t19.npy = IT activations at the last of 20
                            timesteps. float16 by default.
    manifest.json           row order (copy of the image cache's paths),
                            target checkpoint fingerprint, list of files with
                            shapes/dtypes/stats, pseudo-label vs GT agreement.

Row i of every array = row i of the image cache = manifest["paths"][i].

Incremental use
---------------
Running again into the SAME --out_dir with other --layers/--timesteps adds the
new (layer, timestep) files and keeps the existing ones, after checking that
the image cache, checkpoint and settings are the same. Already-present files
are skipped unless --overwrite.

Dry run
-------
--dry_run pushes a couple of batches through the target, writes NOTHING, and
prints the real per-layer shapes, the projected file sizes for the whole
cache (and for the train+val part), the fraction of exact zeros, the
projected run time, and the free disk space.

Usage (from DiCarlo_experiment/):
    python round4/prepare_targets.py \
        --image_cache /scratch/wz2723/dynvision/cache/imagenette_images \
        --target_run  /scratch/wz2723/dynvision/runs/round4/target_x8 \
        --out_dir     /scratch/wz2723/dynvision/cache/targets/target_x8 \
        --layers V1 V4 IT --timesteps -1 --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROUND4_DIR = Path(__file__).resolve().parent
# round4/ first, then DiCarlo_experiment/ (for utils/).
for _p in (ROUND4_DIR.parent, ROUND4_DIR):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from train_target import load_trained_target, read_model_card
from utils.activation_utils import capture_layer_responses, disable_response_storage

CACHE_FORMAT_VERSION = 1
FP16_MAX = 65504.0

logger = logging.getLogger("prepare_targets")


# =============================================================================
# Small helpers
# =============================================================================

def file_sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def activation_key(layer: str, t: int) -> str:
    return f"{layer}_t{t:02d}"


def free_disk_gb(path: Path) -> float:
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def autocast_context(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("--precision must be float32 or bfloat16")


def load_image_cache(cache_dir: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found (image cache missing or incomplete).")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("kind") != "image_cache":
        raise ValueError(f"{manifest_path} is not an image cache manifest.")
    images = np.load(Path(cache_dir) / manifest["array_file"], mmap_mode="r")
    if list(images.shape) != manifest["array_shape"]:
        raise ValueError("images_uint8.npy shape does not match its manifest.")
    return manifest, images


# =============================================================================
# Target forward on one block of rows
# =============================================================================

class Preprocess:
    """uint8 rows -> normalized [B, T, 3, H, W] on the device."""

    def __init__(self, image_manifest: Dict[str, Any], device: torch.device, n_timesteps: int):
        self.mean = torch.tensor(image_manifest["normalize_mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(image_manifest["normalize_std"], device=device).view(1, 3, 1, 1)
        self.device = device
        self.n_timesteps = n_timesteps

    def __call__(self, rows_uint8: np.ndarray) -> torch.Tensor:
        # np.array copies the read-only memmap rows into a normal writable array.
        x = torch.from_numpy(np.array(rows_uint8)).to(self.device, non_blocking=True)
        x = (x.float() / 255.0 - self.mean) / self.std
        # Same static image at every timestep (as ensure_temporal_input does).
        return x.unsqueeze(1).expand(-1, self.n_timesteps, -1, -1, -1)


@torch.no_grad()
def run_target(model, x, layers, timesteps, device, precision):
    with autocast_context(device, precision):
        logits, acts = capture_layer_responses(
            model, x, layers=layers, reset_state=True, timesteps=timesteps,
        )
    if hasattr(model, "reset"):
        model.reset()  # free the recurrent state right away
    return logits.float(), acts


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_cache", required=True, help="Folder made by prepare_images.py.")
    p.add_argument("--out_dir", required=True, help="Folder for this target's cache.")
    p.add_argument("--target_run", required=True,
                   help="Trained target run folder (model_card.yaml + checkpoints/), from round4/train_target.py.")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint file. Default: the one the model card names (checkpoints.target_checkpoint).")
    p.add_argument("--layers", nargs="+", default=["V1", "V4", "IT"])
    p.add_argument("--timesteps", nargs="+", type=int, default=[-1],
                   help="Input timesteps to store (negative = from the end). -1 = last.")
    p.add_argument("--input_timesteps", type=int, default=None,
                   help="Default: the target's n_timesteps from its model card.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32",
                   help="Target forward precision. One-off cost, so float32 by default.")
    p.add_argument("--act_dtype", choices=["float16", "float32"], default="float16",
                   help="Storage dtype of activations (float32 doubles the size).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true", help="Measure and project sizes; write nothing.")
    p.add_argument("--dry_run_batches", type=int, default=3)
    p.add_argument("--overwrite", action="store_true", help="Recompute files that already exist.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Inputs: image cache + target model.
    # ------------------------------------------------------------------
    image_manifest, images = load_image_cache(Path(args.image_cache))
    n = image_manifest["n_images"]
    paths = image_manifest["paths"]
    gt_labels = np.asarray(image_manifest["labels"])
    in_official_train = np.array([p.startswith("train/") for p in paths])
    logger.info("Image cache: %d images from %s", n, args.image_cache)

    card, card_path = read_model_card(args.target_run)
    if card.get("status") != "trained":
        raise RuntimeError(f"{card_path}: status={card.get('status')!r}, target not finished.")
    if args.checkpoint is None:
        args.checkpoint = str(card_path.parent / "checkpoints" / card["checkpoints"]["target_checkpoint"])
    if args.input_timesteps is None:
        args.input_timesteps = int(card["model"]["kwargs"]["n_timesteps"])
    logger.info("Target: %s (%s) | checkpoint %s", card.get("name"), card["model"]["class"], args.checkpoint)

    layers = list(dict.fromkeys(args.layers))
    t_resolved = sorted({t + args.input_timesteps if t < 0 else t for t in args.timesteps})
    if any(not 0 <= t < args.input_timesteps for t in t_resolved):
        raise ValueError(f"--timesteps {args.timesteps} out of range for {args.input_timesteps} timesteps.")

    checkpoint_sha = file_sha256_16(Path(args.checkpoint))

    target_settings = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_16": checkpoint_sha,
        "model_class": card["model"]["class"],
        "model_card": "target_model_card.yaml",
        "input_timesteps": args.input_timesteps,
        "precision": args.precision,
    }

    # ------------------------------------------------------------------
    # Existing cache? -> must be compatible; work out what is still missing.
    # ------------------------------------------------------------------
    existing: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        problems = []
        if existing["image_cache"]["paths_sha256_16"] != image_manifest["paths_sha256_16"]:
            problems.append("built from a different image cache (row order differs)")
        for key, value in target_settings.items():
            if existing["target"].get(key) != value:
                problems.append(f"target.{key}: existing={existing['target'].get(key)!r} now={value!r}")
        if problems:
            raise RuntimeError(
                f"{manifest_path} is incompatible with this run:\n  - " + "\n  - ".join(problems)
                + "\nUse a new --out_dir for a different target/setting."
            )

    have_acts = set(existing["activations"]) if existing else set()
    todo = [(layer, t) for layer in layers for t in t_resolved
            if args.overwrite or activation_key(layer, t) not in have_acts]
    write_logits = args.overwrite or existing is None

    if not args.dry_run:
        skipped = [activation_key(l, t) for l in layers for t in t_resolved if (l, t) not in todo]
        if skipped:
            logger.info("Already cached, skipped: %s", ", ".join(skipped))
        if not todo and not write_logits:
            logger.info("Nothing to do.")
            return

    # Layers/timesteps actually captured in this pass.
    cap_layers = list(dict.fromkeys(l for l, _ in todo)) or layers[:1]
    cap_timesteps = sorted({t for _, t in todo}) or [t_resolved[-1]]

    model = load_trained_target(args.target_run, args.checkpoint, device=device, freeze=True)
    model.eval()
    disable_response_storage(model)
    preprocess = Preprocess(image_manifest, device, args.input_timesteps)

    # ------------------------------------------------------------------
    # Dry run: measure, project, write nothing.
    # ------------------------------------------------------------------
    if args.dry_run:
        bytes_per_value = 2 if args.act_dtype == "float16" else 4
        n_batches = min(args.dry_run_batches, -(-n // args.batch_size))
        zero_frac: Dict[str, List[float]] = {}
        max_abs: Dict[str, float] = {}
        shapes: Dict[str, Tuple[int, ...]] = {}
        logits_shape = None

        times = []
        for b in range(n_batches):
            i, j = b * args.batch_size, min((b + 1) * args.batch_size, n)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            logits, acts = run_target(model, preprocess(images[i:j]), tuple(cap_layers),
                                      tuple(cap_timesteps), device, args.precision)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) / (j - i))
            logits_shape = tuple(logits.shape[1:])
            for layer in cap_layers:
                a = acts[layer].float()
                shapes[layer] = tuple(a.shape[2:])
                zero_frac.setdefault(layer, []).append((a == 0).float().mean().item())
                max_abs[layer] = max(max_abs.get(layer, 0.0), a.abs().max().item())

        sec_per_img = float(np.median(times[1:] if len(times) > 1 else times))  # first batch includes cuDNN warm-up
        n_trainval = int(in_official_train.sum())

        logger.info("=" * 78)
        logger.info("DRY RUN (nothing written). Measured on %d batch(es) of %d.", n_batches, args.batch_size)
        logger.info("logits per image: %s float32 -> %.1f MB for all %d images",
                    logits_shape, np.prod(logits_shape) * 4 * n / 1e6, n)
        logger.info("%-8s %-18s %10s %12s %14s %9s %10s", "file", "shape [C,H,W]", "MB/image",
                    f"all {n}", f"train+val {n_trainval}", "zeros", "max|x|")
        total_all = total_tv = 0.0
        for layer, t in todo:
            values = int(np.prod(shapes[layer]))
            per_img = values * bytes_per_value
            all_gb, tv_gb = per_img * n / 1e9, per_img * n_trainval / 1e9
            total_all += all_gb
            total_tv += tv_gb
            logger.info("%-8s %-18s %10.2f %10.1f GB %12.1f GB %8.1f%% %10.1f",
                        activation_key(layer, t), str(shapes[layer]), per_img / 1e6, all_gb, tv_gb,
                        100 * float(np.mean(zero_frac[layer])), max_abs[layer])
        logger.info("TOTAL (%s): all images %.1f GB | train+val part %.1f GB", args.act_dtype, total_all, total_tv)
        logger.info("Projected target time: %.1f min (%.2f ms/image)", sec_per_img * n / 60, sec_per_img * 1e3)
        logger.info("Free disk at %s: %.0f GB", out_dir, free_disk_gb(out_dir))
        if args.act_dtype == "float16" and max(max_abs.values()) > FP16_MAX:
            logger.warning("Some activations exceed the float16 range -> use --act_dtype float32.")
        return

    # ------------------------------------------------------------------
    # Full run.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(card_path, out_dir / "target_model_card.yaml")
    act_np_dtype = np.float16 if args.act_dtype == "float16" else np.float32

    writers: Dict[str, np.memmap] = {}
    stats: Dict[str, Dict[str, float]] = {}
    logits_mm = None  # created after the first batch, when its shape is known
    old_logits = None
    if not write_logits:
        old_logits = np.load(out_dir / existing["logits"]["file"], mmap_mode="r")

    pseudo = np.empty(n, dtype=np.int64)
    start = time.time()
    for b, i in enumerate(range(0, n, args.batch_size)):
        j = min(i + args.batch_size, n)
        logits, acts = run_target(model, preprocess(images[i:j]), tuple(cap_layers),
                                  tuple(cap_timesteps), device, args.precision)
        logits_np = logits.cpu().numpy()
        pseudo[i:j] = logits_np[:, -1].argmax(-1)

        if write_logits:
            if logits_mm is None:
                logits_mm = np.lib.format.open_memmap(
                    out_dir / "logits.npy", mode="w+", dtype=np.float32, shape=(n, *logits_np.shape[1:]))
            logits_mm[i:j] = logits_np
        elif b == 0:
            # Guard: the stored logits must come from the same model.
            diff = np.abs(np.asarray(old_logits[i:j]) - logits_np).max()
            logger.info("Consistency check vs existing logits.npy: max |diff| = %.3g", diff)
            if diff > 1e-2:
                raise RuntimeError("Existing logits.npy disagrees with this target forward.")

        for layer, t in todo:
            key = activation_key(layer, t)
            a = acts[layer][:, cap_timesteps.index(t)].float()
            if key not in writers:
                writers[key] = np.lib.format.open_memmap(
                    out_dir / f"{key}.npy.partial", mode="w+", dtype=act_np_dtype, shape=(n, *a.shape[1:]))
                stats[key] = {"max_abs": 0.0, "zero_count": 0.0, "sum": 0.0}
            s = stats[key]
            s["max_abs"] = max(s["max_abs"], a.abs().max().item())
            s["zero_count"] += (a == 0).sum().item()
            s["sum"] += a.sum().item()
            if args.act_dtype == "float16" and s["max_abs"] > FP16_MAX:
                raise RuntimeError(f"{key}: values exceed the float16 range; rerun with --act_dtype float32.")
            writers[key][i:j] = a.to(torch.float16 if args.act_dtype == "float16" else torch.float32).cpu().numpy()
        del acts, logits

        if (b + 1) % 50 == 0 or j == n:
            logger.info("  %d / %d images (%.0f s)", j, n, time.time() - start)

    # Flush, then move .partial -> final names (only complete files get final names).
    if logits_mm is not None:
        logits_mm.flush()
    files_info: Dict[str, Any] = dict(existing["activations"]) if existing else {}
    for (layer, t) in todo:
        key = activation_key(layer, t)
        mm = writers.pop(key)
        shape = list(mm.shape)
        mm.flush()
        del mm
        (out_dir / f"{key}.npy.partial").replace(out_dir / f"{key}.npy")
        s = stats[key]
        n_values = float(np.prod(shape))
        files_info[key] = {
            "file": f"{key}.npy",
            "layer": layer,
            "timestep": t,
            "shape": shape,
            "dtype": args.act_dtype,
            "size_gb": round((out_dir / f"{key}.npy").stat().st_size / 1e9, 3),
            "max_abs": s["max_abs"],
            "zero_fraction": s["zero_count"] / n_values,
            "mean": s["sum"] / n_values,
        }

    # Pseudo-label (final prediction) vs ground truth.
    agree = pseudo == gt_labels
    agreement = {
        "official_train_folder": float(agree[in_official_train].mean()) if in_official_train.any() else None,
        "official_val_folder": float(agree[~in_official_train].mean()) if (~in_official_train).any() else None,
    }
    logger.info("Pseudo-label (last timestep) == GT: official train/ %.4f | official val/ %.4f",
                agreement["official_train_folder"] or float("nan"), agreement["official_val_folder"] or float("nan"))

    history = list(existing["history"]) if existing else []
    history.append({
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "wrote": (["logits"] if write_logits else []) + [activation_key(l, t) for l, t in todo],
        "seconds": round(time.time() - start, 1),
    })

    logits_info = existing["logits"] if (existing and not write_logits) else {
        "file": "logits.npy",
        "shape": [n, *np.load(out_dir / "logits.npy", mmap_mode="r").shape[1:]],
        "dtype": "float32",
        "pseudo_label_rule": "argmax(logits[:, -1])",
    }

    manifest = {
        "kind": "target_cache",
        "format_version": CACHE_FORMAT_VERSION,
        "image_cache": {
            "dir": str(Path(args.image_cache).resolve()),
            "paths_sha256_16": image_manifest["paths_sha256_16"],
            "n_images": n,
        },
        "target": target_settings,
        "logits": logits_info,
        "activations": files_info,
        "pseudo_label_vs_gt": agreement,
        "history": history,
        "paths": paths,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    tmp.replace(manifest_path)

    total = sum(v["size_gb"] for v in files_info.values())
    logger.info("Done in %.1f min. Activation files in cache: %s (%.1f GB).",
                (time.time() - start) / 60, ", ".join(sorted(files_info)), total)


if __name__ == "__main__":
    main()
