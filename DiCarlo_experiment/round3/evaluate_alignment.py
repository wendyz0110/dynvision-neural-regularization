"""
Evaluate a set of round 3 train_student.py runs against the frozen target on
the TEST split, in one job. Round 3 copy of DiCarlo_experiment/evaluate_alignment.py.

Round 3 use: ONE job per seed (control + every aligned condition of that seed,
which share the init file -> paired comparison), e.g.
    python round3/evaluate_alignment.py --out_dir .../runs/round3/eval/seed2723 \
        --runs .../students/control_s2723 .../students/IT_w1_s2723 ...
Differences from round 2: imports the round 3 train_student; --out_dir is
the natural way to name the per-seed folders; --unpaired allows runs with
DIFFERENT init files (e.g. the three seeds' controls together: their CKA to
each other is the seed-to-seed baseline).

Each run directory holds ONE student. The runs passed together must share
the same caches, split and student architecture (checked). One of them is
normally the control (alignment.layers == []).

What it computes (numbers only; plotting is in the round-2 notebook)
--------------------------------------------------------------------
For the target and every run's student checkpoint (default: last.pt):

1. metrics_over_time.csv   one row per (model, timestep):
     gt_accuracy            argmax == ground-truth label
     agreement              argmax == target's pseudo-label
                            (target's final prediction, from the target cache,
                            i.e. exactly the label the students trained on;
                            for the target itself this shows how its earlier
                            timesteps agree with its own final answer)
     ce_pseudo              cross-entropy vs that pseudo-label

2. cka_over_time.csv       one row per (layer, timestep, model pair):
     cka_biased, cka_unbiased   linear CKA over the FULL test set at once
     Pairs: every student vs target, and every student vs every other
     student. The notebook plots the biased value vs target; the rest is
     kept for reference.

3. summary.json            headline numbers at the last timestep
   eval_info.json          which runs / checkpoints / caches were used
   logits.npz              [N, T, C] logits of every model + labels
   eval.log

How the full-test-set CKA is computed (exact, no approximation)
---------------------------------------------------------------
CKA only needs each model's own Gram matrix K = X X^T over all N test images
(N x N, ~120 MB), never two models' features at the same time. So for each
group of (layer, timesteps) the script runs ONE model over the test set,
keeps that model's features for the group on the GPU, turns them into Gram
matrices (float64 accumulation), frees the features, and moves on to the
next model. Peak memory = one model's features for one group (V1: 3925
images x 1.8 M features ~ 14 GB per timestep in bf16), so it fits on a 40 GB
A100. Group sizes come from free memory.

Usage (from DiCarlo_experiment/):
    python evaluate_alignment.py --runs \\
        /scratch/wz2723/dynvision/runs/round2/control \\
        /scratch/wz2723/dynvision/runs/round2/aligned_IT_w1 ...
Results go to <parent of the first run>/eval/<date-time>/ unless --out_dir.
Timesteps are 0-based (0..T-1), as in the training code.
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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# round3/ must come BEFORE DiCarlo_experiment/ (both have a train_student.py).
ROUND3_DIR = Path(__file__).resolve().parent
for _p in (ROUND3_DIR.parent, ROUND3_DIR):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

from train_student import (
    ExperimentConfig,
    autocast_context,
    file_sha256_16,
    import_object,
    load_experiment_config,
    student_spec,
)
from utils.activation_utils import capture_layer_responses, disable_response_storage
from utils.cache_data import CachedLoader, CachedSplit, CacheIndex, to_model_input
from utils.data_utils import load_split
from utils.student_model_utils import build_student_model, initialize_lazy_connections
from utils.target_model_utils import load_target_model

logger = logging.getLogger("alignment_eval")

TARGET = "target"


# =============================================================================
# CLI / setup
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories of train_student.py (one student each).")
    p.add_argument("--checkpoint", default="last", choices=("last", "best"),
                   help="Student checkpoint to evaluate (default: last).")
    p.add_argument("--out_dir", default=None,
                   help="Default: <parent of first run>/eval/<date-time>.")
    p.add_argument("--layers", nargs="+", default=["V1", "V4", "IT"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default=None,
                   help="Forward precision. Default: runtime.precision of the first run.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Evaluation batch size (forward only, no gradients).")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--memory_fraction", type=float, default=0.55,
                   help="Fraction of free GPU memory the stored features may use.")
    p.add_argument("--max_grams_per_group", type=int, default=16,
                   help="Cap on timesteps per pass (bounds Gram memory on the CPU).")
    p.add_argument("--unpaired", action="store_true",
                   help="Allow runs with different student init files (different seeds).")
    p.add_argument("--max_test_images", type=int, default=None,
                   help="Use only the first N test images (quick checks only).")
    return p.parse_args(argv)


def setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Runs and models
# =============================================================================

def load_run(run_dir: Path, checkpoint: str) -> Dict:
    cfg = load_experiment_config(run_dir / "resolved_config.yaml")
    info = json.loads((run_dir / "run_info.json").read_text())
    ckpt = run_dir / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found (training not finished?).")
    return {
        "name": run_dir.name,
        "dir": run_dir,
        "cfg": cfg,
        "info": info,
        "checkpoint": ckpt,
        "loss_layers": [l["key"] for l in info.get("loss_layers", [])],
        "loss_weights": {l["key"]: l["weight"] for l in info.get("loss_layers", [])},
    }


def check_runs_compatible(runs: List[Dict], *, paired: bool = True) -> None:
    """Everything that must be identical for the comparison to be fair."""
    ref = runs[0]
    for r in runs[1:]:
        problems = []
        for key in ("image_cache", "target_cache", "paths_sha256_16", "target_checkpoint_sha256_16"):
            if r["info"]["data"].get(key) != ref["info"]["data"].get(key):
                problems.append(f"data.{key}")
        if student_spec(r["cfg"]) != student_spec(ref["cfg"]):
            problems.append("student architecture")
        if paired and r["info"]["student_init"]["sha256_16"] != ref["info"]["student_init"]["sha256_16"]:
            problems.append("student init file")
        if r["cfg"].data.input_timesteps != ref["cfg"].data.input_timesteps:
            problems.append("input_timesteps")
        if problems:
            raise RuntimeError(f"Run {r['name']} differs from {ref['name']} in: {problems}")
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        raise RuntimeError(f"Run directory names must be unique (used as model names): {names}")


def load_student(cfg: ExperimentConfig, path: Path, device: torch.device) -> Tuple[torch.nn.Module, int]:
    """Rebuild the student architecture exactly as in training, then load weights."""
    s = cfg.student
    model = build_student_model(
        model_cls=import_object(s.model_class),
        config_path=s.config_path,
        overrides=s.overrides,
        use_current_defaults=s.use_current_defaults,
        device=device,
    )
    if s.initialize_lazy:
        initialize_lazy_connections(model, probe_shape=tuple(s.probe_shape), device=device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, int(payload["epoch"])


def prepare_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    disable_response_storage(model)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# CKA from Gram matrices (exact; same estimators as round 1)
# =============================================================================

def gram_matrix(features: torch.Tensor, *, chunk: int = 8192) -> torch.Tensor:
    """K = X X^T for X [N, P]; float32 products (TF32 off), float64 accumulation."""
    n, p = features.shape
    gram = torch.zeros((n, n), dtype=torch.float64, device=features.device)
    for start in range(0, p, chunk):
        block = features[:, start:start + chunk].float()
        gram += (block @ block.T).double()
    return gram


def _center(gram: torch.Tensor) -> torch.Tensor:
    return gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()


def cka_biased(k: torch.Tensor, l: torch.Tensor) -> float:
    """Standard linear CKA, the same formula as the training loss."""
    kc, lc = _center(k), _center(l)
    return float((kc * lc).sum() / (torch.linalg.norm(kc) * torch.linalg.norm(lc)))


def _hsic_unbiased(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
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
    """Debiased linear CKA (Song et al. 2012); kept in the CSV for reference."""
    if k.shape[0] < 4:
        return math.nan
    return float(_hsic_unbiased(k, l) / torch.sqrt(_hsic_unbiased(k, k) * _hsic_unbiased(l, l)))


# =============================================================================
# One model, one group of (layer, timesteps): features -> Gram matrices
# =============================================================================

def free_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


@torch.no_grad()
def feature_sizes(model, loader: CachedLoader, layers, *, mean_std, n_timesteps, device, precision) -> Dict[str, int]:
    """Number of features per image of each layer (from one small batch)."""
    batch = next(iter(loader))
    x = to_model_input(batch["images"][:2].to(device), mean=mean_std[0], std=mean_std[1], n_timesteps=n_timesteps)
    with autocast_context(device, precision):
        _, acts = capture_layer_responses(model, x, layers=tuple(layers), timesteps=(-1,))
    if hasattr(model, "reset"):
        model.reset()
    return {layer: int(acts[layer][:, 0].flatten(start_dim=1).shape[1]) for layer in layers}


@torch.no_grad()
def model_pass(
    model: torch.nn.Module,
    loader: CachedLoader,
    *,
    layer: str,
    timesteps: Sequence[int],
    n_images: int,
    n_features: int,
    mean_std,
    n_timesteps: int,
    device: torch.device,
    precision: str,
    keep_logits: bool,
) -> Tuple[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """
    One pass of ONE model over the test set.
    Returns {timestep: Gram [N, N] float64 on CPU} and, if asked, logits [N, T, C].
    """
    feats = {t: torch.empty((n_images, n_features), dtype=torch.bfloat16, device=device) for t in timesteps}
    logits: List[torch.Tensor] = []
    offset = 0
    for batch in loader:
        if offset >= n_images:
            break
        take = min(batch["images"].shape[0], n_images - offset)
        x = to_model_input(batch["images"][:take].to(device, non_blocking=True),
                           mean=mean_std[0], std=mean_std[1], n_timesteps=n_timesteps)
        with autocast_context(device, precision):
            out, acts = capture_layer_responses(model, x, layers=(layer,), reset_state=True,
                                                timesteps=tuple(timesteps))
        for i, t in enumerate(timesteps):
            feats[t][offset:offset + take] = acts[layer][:, i].flatten(start_dim=1).to(torch.bfloat16)
        if keep_logits:
            logits.append(out.float().cpu())
        del acts, out
        if hasattr(model, "reset"):
            model.reset()
        offset += take
    if offset != n_images:
        raise RuntimeError(f"Collected {offset} images, expected {n_images}.")

    grams = {}
    for t in timesteps:
        grams[t] = gram_matrix(feats[t]).cpu()
        del feats[t]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return grams, (torch.cat(logits) if keep_logits else None)


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = parse_args(argv)
    started = time.time()

    runs = [load_run(Path(d), args.checkpoint) for d in args.runs]
    check_runs_compatible(runs, paired=not args.unpaired)
    ref_cfg = runs[0]["cfg"]
    n_timesteps = ref_cfg.data.input_timesteps

    out_dir = Path(args.out_dir) if args.out_dir else (
        runs[0]["dir"].parent / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite earlier results
    setup_logging(out_dir / "eval.log")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    precision = args.precision or ref_cfg.runtime.precision
    torch.backends.cuda.matmul.allow_tf32 = False   # exact float32 Gram products
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    logger.info("Runs: %s", [r["name"] for r in runs])
    logger.info("Checkpoint: %s | device %s | precision %s | out: %s", args.checkpoint, device, precision, out_dir)

    # ---- test data from the caches ----------------------------------------
    index = CacheIndex(ref_cfg.cache.image_dir, ref_cfg.cache.target_dir)
    split = load_split(runs[0]["dir"] / "data" / "split.json")
    for r in runs[1:]:
        if load_split(r["dir"] / "data" / "split.json")["test"] != split["test"]:
            raise RuntimeError(f"Run {r['name']} has a different test split.")
    items = split["test"][: args.max_test_images] if args.max_test_images else split["test"]
    test = CachedSplit(index, items, name="test", target_keys=[],
                       pseudo_label_timestep=ref_cfg.alignment.pseudo_label_timestep)
    loader = CachedLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers)
    n_images = len(test)
    mean_std = (index.mean, index.std)
    pseudo = torch.from_numpy(test.pseudo)
    gt = torch.from_numpy(test.gt)
    logger.info("Test images: %d | pseudo-label (target final prediction, cache) == GT on %.4f",
                n_images, test.pseudo_gt_agreement())

    # ---- models ------------------------------------------------------------
    tgt_info = index.target_manifest["target"]
    tgt_ckpt = None if tgt_info["checkpoint"] == "load_target_model default" else tgt_info["checkpoint"]
    if tgt_ckpt and tgt_info.get("checkpoint_sha256_16") and file_sha256_16(Path(tgt_ckpt)) != tgt_info["checkpoint_sha256_16"]:
        raise RuntimeError(f"Target checkpoint {tgt_ckpt} differs from the one the target cache was built with.")
    models: Dict[str, torch.nn.Module] = {TARGET: prepare_for_eval(
        load_target_model(checkpoint_path=tgt_ckpt, device=device, freeze=True))}
    epochs: Dict[str, int] = {}
    for r in runs:
        model, epoch = load_student(r["cfg"], r["checkpoint"], device)
        models[r["name"]] = prepare_for_eval(model)
        epochs[r["name"]] = epoch
    logger.info("Student checkpoints (%s): %s", args.checkpoint, epochs)

    # ---- plan the passes -------------------------------------------------------
    sizes = feature_sizes(models[TARGET], loader, args.layers, mean_std=mean_std,
                          n_timesteps=n_timesteps, device=device, precision=precision)
    budget = free_bytes(device) * args.memory_fraction
    groups: List[Tuple[str, List[int]]] = []
    for layer in args.layers:
        per_t = n_images * sizes[layer] * 2  # bfloat16 features
        k = int(budget // per_t)
        if k < 1:
            raise MemoryError(f"{layer}: one timestep needs {per_t / 1e9:.1f} GB of features, budget "
                              f"{budget / 1e9:.1f} GB. Lower --batch_size or raise --memory_fraction.")
        k = min(k, args.max_grams_per_group, n_timesteps)
        for s in range(0, n_timesteps, k):
            groups.append((layer, list(range(s, min(s + k, n_timesteps)))))
    logger.info("Features per image: %s | budget %.1f GB | %d group(s) x %d models = %d passes",
                sizes, budget / 1e9, len(groups), len(models), len(groups) * len(models))

    # ---- passes: per group, every model -> Grams -> CKA pairs ------------------
    cka_rows: List[Dict] = []
    logits: Dict[str, torch.Tensor] = {}
    names = list(models)
    for g, (layer, timesteps) in enumerate(groups):
        t0 = time.time()
        grams: Dict[str, Dict[int, torch.Tensor]] = {}
        for name in names:
            grams[name], lg = model_pass(
                models[name], loader, layer=layer, timesteps=timesteps, n_images=n_images,
                n_features=sizes[layer], mean_std=mean_std, n_timesteps=n_timesteps,
                device=device, precision=precision, keep_logits=name not in logits,
            )
            if lg is not None:
                logits[name] = lg
        for t in timesteps:
            for a, b in combinations(names, 2):
                if a == TARGET:          # always: student first, target second
                    a, b = b, a
                cka_rows.append({
                    "layer": layer, "timestep": t, "model_a": a, "model_b": b,
                    "cka_biased": cka_biased(grams[a][t], grams[b][t]),
                    "cka_unbiased": cka_unbiased(grams[a][t], grams[b][t]),
                })
        del grams
        logger.info("Group %d/%d (%s, t=%d..%d) done in %.0f s", g + 1, len(groups), layer,
                    timesteps[0], timesteps[-1], time.time() - t0)
    write_csv(out_dir / "cka_over_time.csv", cka_rows)

    # ---- behavior from the logits ----------------------------------------------
    metric_rows = []
    for name in names:
        lg = logits[name]
        for t in range(lg.shape[1]):
            pred = lg[:, t].argmax(-1)
            metric_rows.append({
                "model": name, "timestep": t,
                "gt_accuracy": (pred == gt).float().mean().item(),
                "agreement": (pred == pseudo).float().mean().item(),
                "ce_pseudo": F.cross_entropy(lg[:, t], pseudo).item(),
            })
    write_csv(out_dir / "metrics_over_time.csv", metric_rows)
    np.savez_compressed(out_dir / "logits.npz", **{n: logits[n].numpy() for n in names},
                        labels=gt.numpy(), pseudo_labels=pseudo.numpy(),
                        sample_paths=np.array([p for p, _ in items]), classes=np.array(split["classes"]))

    # Sanity: online target (this precision) vs the cached target (float32).
    online_final = logits[TARGET][:, pseudo_label_index(ref_cfg)].argmax(-1)
    target_consistency = (online_final == pseudo).float().mean().item()
    logger.info("Online target final prediction == cached pseudo-label on %.4f of test images", target_consistency)
    if target_consistency < 0.98:
        logger.warning("Online target and target cache disagree on >2%% of images: check checkpoint / precision.")

    # ---- summary ----------------------------------------------------------------
    last = n_timesteps - 1

    def m(name, key):
        return next(r[key] for r in metric_rows if r["model"] == name and r["timestep"] == last)

    def c(name, layer):
        return next(r["cka_biased"] for r in cka_rows if r["model_a"] == name and r["model_b"] == TARGET
                    and r["layer"] == layer and r["timestep"] == last)

    summary = {
        "out_dir": str(out_dir),
        "checkpoint": args.checkpoint,
        "n_test_images": n_images,
        "last_timestep": last,
        "target": {"gt_accuracy": m(TARGET, "gt_accuracy"), "online_vs_cache_agreement": target_consistency},
        "students": {
            r["name"]: {
                "epoch": epochs[r["name"]],
                "loss_layers": r["loss_layers"],
                "gt_accuracy": m(r["name"], "gt_accuracy"),
                "agreement": m(r["name"], "agreement"),
                "ce_pseudo": m(r["name"], "ce_pseudo"),
                "cka_biased_vs_target": {layer: c(r["name"], layer) for layer in args.layers},
            }
            for r in runs
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "eval_info.json").open("w") as f:
        json.dump({
            "args": vars(args),
            "paired": not args.unpaired,
            "student_inits": {r["name"]: r["info"]["student_init"] for r in runs},
            "precision": precision,
            "runs": [{"name": r["name"], "dir": str(r["dir"].resolve()), "role": r["cfg"].experiment.role,
                      "description": r["cfg"].experiment.description, "loss_layers": r["loss_layers"],
                      "loss_weights": r["loss_weights"], "is_control": not r["loss_layers"],
                      "checkpoint": str(r["checkpoint"]), "epoch": epochs[r["name"]]} for r in runs],
            "layers": args.layers,
            "ce_first_timestep": ref_cfg.alignment.ce_first_timestep,
            "data": index.fingerprint(),
            "feature_sizes": sizes,
            "torch_version": torch.__version__,
        }, f, indent=2)

    logger.info("Target GT accuracy (t=%d): %.4f", last, summary["target"]["gt_accuracy"])
    for name, s in summary["students"].items():
        logger.info("%-22s gt_acc %.4f | agreement %.4f | ce %.4f | CKA %s", name, s["gt_accuracy"],
                    s["agreement"], s["ce_pseudo"],
                    " ".join(f"{k}={v:.4f}" for k, v in s["cka_biased_vs_target"].items()))
    logger.info("Done in %.1f min. Results in %s", (time.time() - started) / 60, out_dir)
    return summary


def pseudo_label_index(cfg: ExperimentConfig) -> int:
    return cfg.alignment.pseudo_label_timestep


if __name__ == "__main__":
    main()
