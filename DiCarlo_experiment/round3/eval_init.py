"""
Measure every round 3 model BEFORE training (the "epoch 0" point of the
training curves), with exactly the same code and data as during training.

Why: metrics.csv row "epoch 0" is measured AFTER the first training epoch.
This script adds the missing starting point, so the curves can start at
"0 epochs completed" (= initial weights).

Students: for each run, the student is rebuilt from its init file
(student.init_checkpoint) and run_epoch(split="val") of round3/train_student.py
is called with the run's own loss terms and monitored layers -- the same
function that produced the val_* columns of metrics.csv. The gradient
diagnostics (diag_* columns) are computed on the same fixed batch as in
training. Runs of one seed share the init file, so their CE / agreement /
CKA at epoch 0 are identical by construction (only val_total_loss and
val_align_loss differ: different loss terms).

Target: rebuilt with round3/train_target.py build_model() from its YAML
(same seed -> same initial weights) and evaluated with its evaluate() on
val and on train (eval mode), i.e. the val_* and train_eval_* columns.

Output (never overwrites training files):
    <run dir>/metrics/init_metrics.csv      one row, same column names as
                                            metrics.csv, epoch = -1
    <target run dir>/metrics/init_metrics.csv

Usage (from DiCarlo_experiment/):
    python round3/eval_init.py --configs round3/configs/students/*.yaml \\
                               --target_config round3/configs/target_r3.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict

ROUND3_DIR = Path(__file__).resolve().parent
for _p in (ROUND3_DIR.parent, ROUND3_DIR):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import numpy as np
import torch

import train_student as ts
import train_target as tt
from utils.activation_utils import disable_response_storage
from utils.cache_data import CachedLoader, CachedSplit, CacheIndex, read_batch
from utils.data_utils import load_split

logger = logging.getLogger("eval_init")


def write_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def student_init_row(config_path: str, device: torch.device) -> Path:
    cfg = ts.load_experiment_config(config_path)
    run_dir = Path(cfg.runtime.output_dir)
    if not (run_dir / "metrics" / "metrics.csv").exists():
        raise FileNotFoundError(f"{run_dir}: no metrics.csv (run not trained?)")
    ts.set_global_seed(cfg.runtime.seed)

    a, T = cfg.alignment, cfg.data.input_timesteps
    align_specs = ts.parse_layer_specs(a.layers, T, "alignment.layers")
    align_keys = {s.key for s in align_specs}
    monitor_specs = [s for s in ts.parse_layer_specs(a.monitor_layers, T, "alignment.monitor_layers")
                     if s.key not in align_keys]
    index = CacheIndex(cfg.cache.image_dir, cfg.cache.target_dir)
    keys = [s.key for s in align_specs + monitor_specs]
    index.require(keys, "init evaluation")

    split = load_split(Path(cfg.data.split_file))
    train_split = CachedSplit(index, split["train"], name="train", target_keys=keys,
                              pseudo_label_timestep=a.pseudo_label_timestep)
    val_split = CachedSplit(index, split["val"], name="val", target_keys=keys,
                            pseudo_label_timestep=a.pseudo_label_timestep)
    val_loader = CachedLoader(val_split, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False,
                              num_workers=cfg.data.num_workers)
    mean_std = (index.mean, index.std)

    init = torch.load(cfg.student.init_checkpoint, map_location="cpu", weights_only=False)
    if init.get("student_spec") != ts.student_spec(cfg):
        raise RuntimeError(f"{config_path}: student spec differs from {cfg.student.init_checkpoint}")
    model = ts.build_student(cfg, device, seed=int(init["seed"]))
    model.load_state_dict(init["model_state_dict"], strict=True)
    disable_response_storage(model)

    # epoch=0 for run_epoch: a negative epoch would count as "warmup" (CKA terms
    # off). The row is still labelled epoch -1 (= before training).
    val_m = ts.run_epoch(split="val", epoch=0, loader=val_loader, model=model, optimizer=None,
                         align_specs=align_specs, monitor_specs=monitor_specs, cfg=cfg,
                         mean_std=mean_std, device=device, logger=logger)
    row: Dict[str, Any] = {"epoch": -1}
    row.update({f"val_{k}": v for k, v in val_m.items()})

    if cfg.diagnostics.gradient_terms:
        n_diag = min(int(cfg.diagnostics.batch_size or cfg.data.batch_size), len(train_split))
        g = torch.Generator().manual_seed(int(cfg.diagnostics.seed))
        positions = np.sort(torch.randperm(len(train_split), generator=g)[:n_diag].numpy())
        diag = ts.gradient_diagnostics(model=model, batch=read_batch(train_split, positions),
                                       align_specs=align_specs, cfg=cfg, mean_std=mean_std, device=device)
        row.update(diag)

    out = run_dir / "metrics" / "init_metrics.csv"
    write_row(out, row)
    logger.info("%-26s val_ce=%.4f agree=%.4f gt_acc=%.4f | %s", run_dir.name, row["val_ce_loss"],
                row["val_target_agreement"], row["val_gt_accuracy"],
                " ".join(f"{k[8:]}={v:.4f}" for k, v in row.items() if k.startswith("val_cka_")))
    return out


def target_init_row(config_path: str, device: torch.device) -> Path:
    cfg = tt.load_config(config_path)
    run_dir = Path(cfg.runtime.output_dir)
    if not (run_dir / "metrics" / "metrics.csv").exists():
        raise FileNotFoundError(f"{run_dir}: no metrics.csv (target not trained?)")
    import json
    image_dir = Path(cfg.data.image_dir)
    manifest = json.loads((image_dir / "manifest.json").read_text())
    split = load_split(Path(cfg.data.split_file))
    train_split = tt.ImageCacheSplit(image_dir, split["train"], "train", manifest)
    val_split = tt.ImageCacheSplit(image_dir, split["val"], "val", manifest)
    bs, nw = int(cfg.data.batch_size), int(cfg.data.num_workers)
    mean_std = (tuple(manifest["normalize_mean"]), tuple(manifest["normalize_std"]))
    n_timesteps = int(cfg.model.kwargs["n_timesteps"])

    model = tt.build_model(cfg)            # seeds init_seed first -> the training's initial weights
    disable_response_storage(model)
    model.to(device)
    common = dict(model=model, cfg=cfg, mean_std=mean_std, n_timesteps=n_timesteps, device=device)
    val_m = tt.evaluate(loader=tt.ImageLoader(val_split, batch_size=bs, shuffle=False, drop_last=False,
                                              num_workers=nw), **common)
    train_m = tt.evaluate(loader=tt.ImageLoader(train_split, batch_size=bs, shuffle=False, drop_last=False,
                                                num_workers=nw), **common)
    row: Dict[str, Any] = {"epoch": -1,
                           "train_eval_ce_loss": train_m["ce_loss"],
                           "train_eval_accuracy": train_m["accuracy"]}
    row.update({f"val_{k}": v for k, v in val_m.items()})
    out = run_dir / "metrics" / "init_metrics.csv"
    write_row(out, row)
    logger.info("target %s: val_ce=%.4f val_acc=%.4f train_acc=%.4f", run_dir.name, row["val_ce_loss"],
                row["val_accuracy"], row["train_eval_accuracy"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=[], help="Round 3 student configs.")
    parser.add_argument("--target_config", default=None, help="Round 3 target config.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    torch.backends.cudnn.benchmark = True

    if args.target_config:
        target_init_row(args.target_config, device)
    for path in args.configs:
        student_init_row(path, device)
    logger.info("Done: %d student run(s)%s.", len(args.configs), " + target" if args.target_config else "")


if __name__ == "__main__":
    main()
