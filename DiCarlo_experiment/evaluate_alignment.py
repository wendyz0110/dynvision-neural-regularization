"""
Evaluation for a finished train_alignment.py run.

Reads a training run directory, reloads the frozen target plus the aligned and
control students, runs them on the TEST split (official Imagenette val/
folder, never seen by the target) and writes numeric results only. Plotting
is done separately from these files.

What it computes
----------------
1. Per-timestep classification metrics for target, aligned and control
   (from the saved logits): accuracy vs ground truth, agreement with the
   target's pseudo-label, CE vs ground truth, CE vs pseudo-label.
2. Per-timestep IT-layer CKA on the FULL test set (one CKA per timestep over
   all test images, not a batch average), for aligned-vs-target,
   control-vs-target and aligned-vs-control, both biased (as in the training
   loss / DiCarlo) and unbiased (Song et al. 2012; removes the n << p
   inflation).
3. Training trajectory: CKA at the alignment timestep and final-step
   accuracy for every saved checkpoint (init, checkpoints/epochs/*, last),
   to compare both students at the same training points.

How the full-test-set CKA is computed (exact, no approximation)
---------------------------------------------------------------
CKA needs each model's Gram matrix K = X X^T over all N test images, which
needs every image's IT features at once (~3 GB per model per timestep).
So timesteps are processed in groups: one pass over the test set collects
the features of a few timesteps for all three models, their Gram matrices
are computed (float64 accumulation) and the features are freed before the
next group. The group size is chosen from free memory.

Outputs: <run_dir>/eval/<name>/   (training outputs are only ever read)
    logits.npz                [N, T, C] logits per model + labels + paths
    metrics_over_time.csv     model, timestep, accuracy, agreement, CE ...
    cka_over_time.csv         timestep, pair, cka_biased, cka_unbiased
    training_trajectory.csv   epoch, model, cka_*, accuracy, agreement
    summary.json              headline numbers + run information
    eval_args.json, eval.log

Timesteps are 0-based everywhere (0..T-1), as in the training code.

Usage (from the DiCarlo_experiment folder):
    python evaluate_alignment.py --run_dir /scratch/<netid>/DiCarlo_runs/run_01
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_alignment import (
    ExperimentConfig,
    autocast_context,
    ensure_temporal_input,
    import_object,
    load_experiment_config,
    resolve_device,
    unpack_batch,
)
from utils.activation_utils import capture_layer_responses
from utils.data_utils import build_test_loader, load_split
from utils.student_model_utils import build_student_model, initialize_lazy_connections
from utils.target_model_utils import load_target_model

logger = logging.getLogger("alignment_eval")

MODEL_NAMES = ("target", "aligned", "control")
CKA_PAIRS = (
    ("aligned", "target"),
    ("control", "target"),
    ("aligned", "control"),
)


# =============================================================================
# CLI and setup
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="Output folder of a train_alignment.py run.")
    p.add_argument("--checkpoint", default="last", choices=("last", "best"),
                   help="Which student checkpoints to evaluate (default: last).")
    p.add_argument("--name", default=None,
                   help="Results subfolder name (default: date-time).")

    # Paths that may differ from the training machine.
    p.add_argument("--data_root", default=None,
                   help="Override data.root from the run's config.")
    p.add_argument("--target_checkpoint", default=None,
                   help="Override target.checkpoint_path from the run's config.")

    p.add_argument("--device", default=None,
                   help="Default: runtime.device from the run's config.")
    p.add_argument("--precision", default=None,
                   help="Default: runtime.precision from the run's config.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=None)

    p.add_argument("--timesteps_per_pass", type=int, default=None,
                   help="Timesteps collected per pass for the CKA. Default: "
                        "as many as fit in --memory_fraction of free memory.")
    p.add_argument("--memory_fraction", type=float, default=0.6,
                   help="Fraction of free memory the stored features may use.")

    p.add_argument("--skip_trajectory", action="store_true",
                   help="Skip the per-checkpoint training trajectory.")
    p.add_argument("--max_test_images", type=int, default=None,
                   help="Use only the first N test images (quick checks only).")
    p.add_argument("--expected_target_accuracy", type=float, default=None,
                   help="Target's reported test accuracy (0-1), for the "
                        "pipeline sanity check in summary.json.")
    return p.parse_args(argv)


def setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


# =============================================================================
# Loading models
# =============================================================================

def load_student(cfg: ExperimentConfig, path: Path, device: torch.device) -> torch.nn.Module:
    """Rebuild the student architecture exactly as in training, then load weights."""
    model = build_student_model(
        model_cls=import_object(cfg.student.model_class),
        config_path=cfg.student.config_path,
        overrides=cfg.student.overrides,
        use_current_defaults=cfg.student.use_current_defaults,
        device=device,
    )
    if cfg.student.initialize_lazy:
        initialize_lazy_connections(
            model, probe_shape=tuple(cfg.student.probe_shape), device=device
        )

    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def checkpoint_epoch(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def list_trajectory_checkpoints(ckpt_dir: Path) -> List[Tuple[int, Path, Path]]:
    """(epoch, aligned_path, control_path) for init, periodic and last."""
    found: Dict[int, Tuple[Path, Path]] = {}

    def add(aligned: Path, control: Path) -> None:
        if aligned.exists() and control.exists():
            found[checkpoint_epoch(aligned)] = (aligned, control)

    add(ckpt_dir / "aligned_init.pt", ckpt_dir / "control_init.pt")  # epoch -1
    for aligned in sorted((ckpt_dir / "epochs").glob("aligned_epoch*.pt")):
        add(aligned, aligned.with_name(aligned.name.replace("aligned_", "control_")))
    add(ckpt_dir / "aligned_last.pt", ckpt_dir / "control_last.pt")

    return [(e, a, c) for e, (a, c) in sorted(found.items())]


# =============================================================================
# CKA from Gram matrices
# =============================================================================

def gram_matrix(features: torch.Tensor, *, chunk: int = 8192) -> torch.Tensor:
    """
    K = X X^T for X of shape [N, P], accumulated in float64.

    Works through the feature dimension in chunks, so the [N, P] matrix can
    stay in its storage dtype; each chunk is multiplied in float32 (TF32 off)
    and summed in float64.
    """
    n, p = features.shape
    gram = torch.zeros((n, n), dtype=torch.float64, device=features.device)
    for start in range(0, p, chunk):
        block = features[:, start:start + chunk].float()
        gram += (block @ block.T).double()
    return gram


def _center(gram: torch.Tensor) -> torch.Tensor:
    return (
        gram
        - gram.mean(dim=0, keepdim=True)
        - gram.mean(dim=1, keepdim=True)
        + gram.mean()
    )


def cka_biased(k: torch.Tensor, l: torch.Tensor) -> float:
    """Standard (biased) linear CKA, as used in the training loss."""
    kc, lc = _center(k), _center(l)
    return float((kc * lc).sum() / (torch.linalg.norm(kc) * torch.linalg.norm(lc)))


def _hsic_unbiased(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012), from uncentered Grams."""
    n = k.shape[0]
    k = k.clone()
    l = l.clone()
    k.fill_diagonal_(0)
    l.fill_diagonal_(0)
    ones = torch.ones(n, 1, dtype=k.dtype, device=k.device)
    term1 = (k * l).sum()
    term2 = (ones.T @ k @ ones) * (ones.T @ l @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones.T @ k @ l @ ones)
    return (term1 + term2 - term3).squeeze() / (n * (n - 3))


def cka_unbiased(k: torch.Tensor, l: torch.Tensor) -> float:
    """Debiased linear CKA; ~0 for unrelated representations at any n."""
    if k.shape[0] < 4:
        return math.nan
    hsic_kl = _hsic_unbiased(k, l)
    denom = torch.sqrt(_hsic_unbiased(k, k) * _hsic_unbiased(l, l))
    return float(hsic_kl / denom)


# =============================================================================
# Forward passes over the test set
# =============================================================================

def available_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def probe_feature_shape(
    model: torch.nn.Module, loader, cfg: ExperimentConfig, device, precision: str
) -> Tuple[int, torch.dtype]:
    """Number of features P of the alignment layer, and its dtype."""
    images, _, _ = unpack_batch(next(iter(loader)))
    x = ensure_temporal_input(images[:2].to(device), n_timesteps=cfg.data.input_timesteps)
    with torch.no_grad(), autocast_context(device, precision):
        _, acts = capture_layer_responses(
            model, x, layers=(cfg.alignment.layer,), timesteps=(-1,)
        )
    feat = acts[cfg.alignment.layer][:, 0].flatten(start_dim=1)
    return feat.shape[1], feat.dtype


def collect_pass(
    models: Mapping[str, torch.nn.Module],
    loader,
    cfg: ExperimentConfig,
    device: torch.device,
    precision: str,
    *,
    timesteps: Sequence[int],
    n_images: int,
    n_features: int,
    feature_dtype: torch.dtype,
    keep_logits: bool,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, torch.Tensor], torch.Tensor, List[int]]:
    """
    One pass over the test set.

    Returns:
        features[model][t]  -> [N, P] alignment-layer features at timestep t
        logits[model]       -> [N, T, C] (only if keep_logits)
        labels              -> [N]
        sample_ids          -> positions in the split's test list
    """
    layer = cfg.alignment.layer
    features = {
        name: {
            t: torch.empty((n_images, n_features), dtype=feature_dtype, device=device)
            for t in timesteps
        }
        for name in models
    }
    logits: Dict[str, List[torch.Tensor]] = {name: [] for name in models}
    labels: List[torch.Tensor] = []
    sample_ids: List[int] = []

    offset = 0
    for batch in loader:
        images, gt, ids = unpack_batch(batch)
        if offset >= n_images:
            break
        take = min(images.shape[0], n_images - offset)
        images = images[:take].to(device, non_blocking=True)
        x = ensure_temporal_input(images, n_timesteps=cfg.data.input_timesteps)

        for name, model in models.items():
            with torch.no_grad(), autocast_context(device, precision):
                out, acts = capture_layer_responses(
                    model, x, layers=(layer,), reset_state=True, timesteps=tuple(timesteps)
                )
            for i, t in enumerate(timesteps):
                features[name][t][offset:offset + take] = (
                    acts[layer][:, i].flatten(start_dim=1).to(feature_dtype)
                )
            if keep_logits:
                logits[name].append(out.float().cpu())
            del acts, out

        labels.append(gt[:take].cpu())
        sample_ids += [int(i) for i in ids[:take]]
        offset += take

    if offset != n_images:
        raise RuntimeError(f"Collected {offset} images, expected {n_images}.")

    stacked = {name: torch.cat(v) for name, v in logits.items()} if keep_logits else {}
    return features, stacked, torch.cat(labels), sample_ids


# =============================================================================
# Metrics from logits
# =============================================================================

def metrics_over_time(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    pseudo_label_timestep: int,
) -> List[Dict[str, float]]:
    pseudo = logits["target"][:, pseudo_label_timestep].argmax(dim=-1)
    rows = []
    for name in MODEL_NAMES:
        lg = logits[name]
        for t in range(lg.shape[1]):
            pred = lg[:, t].argmax(dim=-1)
            rows.append({
                "model": name,
                "timestep": t,
                "accuracy": (pred == labels).float().mean().item(),
                "agreement_with_pseudolabel": (pred == pseudo).float().mean().item(),
                "ce_ground_truth": F.cross_entropy(lg[:, t], labels).item(),
                "ce_pseudolabel": F.cross_entropy(lg[:, t], pseudo).item(),
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)

    cfg = load_experiment_config(run_dir / "resolved_config.yaml")
    if args.data_root:
        cfg.data.root = args.data_root
    if args.target_checkpoint:
        cfg.target.checkpoint_path = args.target_checkpoint
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    device = resolve_device(args.device or cfg.runtime.device)
    precision = args.precision or cfg.runtime.precision

    out_dir = run_dir / "eval" / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite earlier results
    setup_logging(out_dir / "eval.log")
    with (out_dir / "eval_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Exact float32 products for the Gram matrices (no TF32 shortcuts).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    started = time.time()
    logger.info("Run dir: %s", run_dir)
    logger.info("Results: %s", out_dir)
    logger.info("Device: %s | precision: %s", device, precision)

    # ---- data ----------------------------------------------------------------
    split = load_split(run_dir / "data" / "split.json")
    loader = build_test_loader(cfg, split, batch_size=args.batch_size)
    n_images = len(loader.dataset)
    if args.max_test_images:
        n_images = min(n_images, args.max_test_images)
    logger.info("Test images: %d (of %d)", n_images, len(loader.dataset))

    # ---- models --------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    models = {
        "target": load_target_model(
            checkpoint_path=cfg.target.checkpoint_path, device=device, freeze=True
        ),
        "aligned": load_student(cfg, ckpt_dir / f"aligned_{args.checkpoint}.pt", device),
        "control": load_student(cfg, ckpt_dir / f"control_{args.checkpoint}.pt", device),
    }
    for m in models.values():
        m.eval()
    student_epoch = checkpoint_epoch(ckpt_dir / f"aligned_{args.checkpoint}.pt")
    logger.info("Students: %s checkpoints (epoch %d)", args.checkpoint, student_epoch)

    # ---- plan the passes -----------------------------------------------------
    n_timesteps = cfg.data.input_timesteps
    n_features, feature_dtype = probe_feature_shape(
        models["target"], loader, cfg, device, precision
    )
    bytes_per_timestep = len(models) * n_images * n_features * torch.finfo(feature_dtype).bits // 8
    if args.timesteps_per_pass:
        per_pass = args.timesteps_per_pass
    else:
        budget = available_bytes(device) * args.memory_fraction
        per_pass = int(budget // bytes_per_timestep)
        if per_pass < 1:
            raise MemoryError(
                f"Not enough free memory for one timestep "
                f"({bytes_per_timestep / 1e9:.1f} GB needed for "
                f"{len(models)} models x {n_images} images x {n_features} "
                "features). Use a larger GPU or --max_test_images."
            )
    per_pass = min(per_pass, n_timesteps)
    groups = [list(range(s, min(s + per_pass, n_timesteps)))
              for s in range(0, n_timesteps, per_pass)]
    logger.info(
        "IT features: %d per image (%s). %.1f GB per timestep for 3 models -> "
        "%d timestep(s) per pass, %d pass(es).",
        n_features, feature_dtype, bytes_per_timestep / 1e9, per_pass, len(groups),
    )

    # ---- per-timestep CKA + logits -------------------------------------------
    cka_rows = []
    logits = labels = sample_ids = None

    for g, timesteps in enumerate(groups):
        t0 = time.time()
        feats, pass_logits, pass_labels, pass_ids = collect_pass(
            models, loader, cfg, device, precision,
            timesteps=timesteps, n_images=n_images, n_features=n_features,
            feature_dtype=feature_dtype, keep_logits=(g == 0),
        )
        if g == 0:
            logits, labels, sample_ids = pass_logits, pass_labels, pass_ids

        for t in timesteps:
            grams = {name: gram_matrix(feats[name][t]) for name in models}
            for name in models:
                del feats[name][t]  # free features as soon as the Gram exists
            for a, b in CKA_PAIRS:
                cka_rows.append({
                    "timestep": t,
                    "pair": f"{a}_vs_{b}",
                    "cka_biased": cka_biased(grams[a], grams[b]),
                    "cka_unbiased": cka_unbiased(grams[a], grams[b]),
                })
            del grams
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Pass %d/%d (timesteps %s) done in %.0f s",
                    g + 1, len(groups), timesteps, time.time() - t0)

    write_csv(out_dir / "cka_over_time.csv", cka_rows)

    # ---- metrics from logits ---------------------------------------------------
    metric_rows = metrics_over_time(logits, labels, cfg.target.pseudo_label_timestep)
    write_csv(out_dir / "metrics_over_time.csv", metric_rows)

    test_paths = [split["test"][i][0] for i in sample_ids]
    np.savez_compressed(
        out_dir / "logits.npz",
        target=logits["target"].numpy(),
        aligned=logits["aligned"].numpy(),
        control=logits["control"].numpy(),
        labels=labels.numpy(),
        sample_paths=np.array(test_paths),
        classes=np.array(split["classes"]),
    )

    # ---- training trajectory -----------------------------------------------------
    trajectory_rows = []
    if not args.skip_trajectory:
        align_t = cfg.alignment.timestep % n_timesteps
        points = list_trajectory_checkpoints(ckpt_dir)
        logger.info("Trajectory: %d checkpoint(s), epochs %s",
                    len(points), [e for e, _, _ in points])

        target_only = {"target": models["target"]}
        feats, target_logits, traj_labels, _ = collect_pass(
            target_only, loader, cfg, device, precision, timesteps=[align_t],
            n_images=n_images, n_features=n_features, feature_dtype=feature_dtype,
            keep_logits=True,
        )
        target_gram = gram_matrix(feats["target"][align_t])
        del feats
        pseudo = target_logits["target"][:, cfg.target.pseudo_label_timestep].argmax(-1)

        for epoch, aligned_path, control_path in points:
            t0 = time.time()
            students = {
                "aligned": load_student(cfg, aligned_path, device),
                "control": load_student(cfg, control_path, device),
            }
            feats, s_logits, _, _ = collect_pass(
                students, loader, cfg, device, precision, timesteps=[align_t],
                n_images=n_images, n_features=n_features,
                feature_dtype=feature_dtype, keep_logits=True,
            )
            for name in students:
                gram = gram_matrix(feats[name][align_t])
                del feats[name][align_t]
                pred = s_logits[name][:, -1].argmax(-1)
                trajectory_rows.append({
                    "epoch": epoch,
                    "model": name,
                    "cka_biased": cka_biased(gram, target_gram),
                    "cka_unbiased": cka_unbiased(gram, target_gram),
                    "accuracy_final_timestep": (pred == traj_labels).float().mean().item(),
                    "agreement_final_timestep": (pred == pseudo).float().mean().item(),
                })
                del gram
            del students, feats
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("Trajectory epoch %d done in %.0f s", epoch, time.time() - t0)

        write_csv(out_dir / "training_trajectory.csv", trajectory_rows)

    # ---- summary -------------------------------------------------------------
    last_t = n_timesteps - 1
    align_t = cfg.alignment.timestep % n_timesteps

    def metric(model: str, key: str, t: int) -> float:
        return next(r[key] for r in metric_rows if r["model"] == model and r["timestep"] == t)

    def cka(pair: str, key: str, t: int) -> float:
        return next(r[key] for r in cka_rows if r["pair"] == pair and r["timestep"] == t)

    target_acc = metric("target", "accuracy", last_t)
    summary = {
        "run_dir": str(run_dir),
        "student_checkpoint": args.checkpoint,
        "student_epoch": student_epoch,
        "n_test_images": n_images,
        "alignment_layer": cfg.alignment.layer,
        "alignment_timestep": align_t,
        "target_sanity_check": {
            "target_test_accuracy_final_timestep": target_acc,
            "expected": args.expected_target_accuracy,
            "difference": (
                None if args.expected_target_accuracy is None
                else target_acc - args.expected_target_accuracy
            ),
        },
        "accuracy_final_timestep": {m: metric(m, "accuracy", last_t) for m in MODEL_NAMES},
        "agreement_with_pseudolabel_final_timestep": {
            m: metric(m, "agreement_with_pseudolabel", last_t) for m in ("aligned", "control")
        },
        "cka_at_alignment_timestep": {
            pair: {
                "biased": cka(pair, "cka_biased", align_t),
                "unbiased": cka(pair, "cka_unbiased", align_t),
            }
            for pair in (f"{a}_vs_{b}" for a, b in CKA_PAIRS)
        },
        "feature_dimension": n_features,
        "passes": len(groups),
        "runtime_seconds": round(time.time() - started, 1),
        "torch_version": torch.__version__,
        "device": str(device),
        "precision": precision,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Target test accuracy (final timestep): %.4f%s", target_acc,
                "" if args.expected_target_accuracy is None
                else f" (expected {args.expected_target_accuracy:.4f})")
    if (args.expected_target_accuracy is not None
            and abs(target_acc - args.expected_target_accuracy) > 0.05):
        logger.warning("Target accuracy differs from the expected value by more "
                       "than 5 points: check preprocessing / DynVision version.")
    for pair, v in summary["cka_at_alignment_timestep"].items():
        logger.info("CKA %s @ t=%d: biased %.4f | unbiased %.4f",
                    pair, align_t, v["biased"], v["unbiased"])
    logger.info("Done in %.0f s. Results in %s", time.time() - started, out_dir)
    return summary


if __name__ == "__main__":
    main()