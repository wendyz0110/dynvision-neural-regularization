"""
Train ONE student model (aligned or control) from the precomputed caches -- ROUND 3.

Round 3 copy of DiCarlo_experiment/train_student.py (round 2, unchanged).
Differences from round 2:
  - optimization.optimizer: "adamw" (decoupled weight decay) or "adam";
    plus optimization.betas / eps. Round 3 uses AdamW, wd 0.01,
    max_grad_norm 500, cosine T_max 200 (same recipe as the round 3 target).
  - <output_dir>/model_card.yaml: one file with the student's full
    configuration (architecture incl. DynVision's effective values, data,
    loss, recipe, init, environment) and, at the end, its final metrics.
  - Importable from DiCarlo_experiment/round3 (utils/ is one level up).
Everything else (loss, caches, diagnostics, resume) is the round 2 code.

Replaces train_alignment.py, which trained target + aligned + control in one
loop. Now:

- The target is NOT loaded. Its pseudo-labels and layer activations come from
  the target cache (prepare_targets.py); the images from the image cache
  (prepare_images.py). No augmentation.
- One run = one student. Aligned vs control is decided by the YAML alone:

      alignment.layers: [IT]            -> CE + w * log(1 - CKA_IT)
      alignment.layers: [V1, V4, IT]    -> CE + sum of three CKA terms
      alignment.layers: []              -> CE only  (= the control)

  Any subset of the cached layers/timesteps works; each entry can carry its
  own timestep and weight. `alignment.monitor_layers` are CKA values that are
  only LOGGED (no gradient), so aligned and control runs report the same
  numbers.
- Runs that are to be compared (control vs aligned_*) must share:
    student.init_checkpoint   (identical starting weights; make_student_init.py)
    data.* split settings     (identical train/val images)
    data.batch_size, data.shuffle_seed  (identical batches, identical order)
  The per-epoch order fingerprint is logged in metrics.csv (train_order) so
  this can be checked.

Loss (unchanged from train_alignment.py):
    CE vs the target's final prediction, averaged over timesteps
    [ce_first_timestep, T)  +  sum_l weight_l * log(1 - CKA_l)
CKA terms are skipped for batches smaller than min_cka_batch_size and during
warmup_epochs (still logged).

Outputs in runtime.output_dir:
    resolved_config.yaml, run_info.json, data/split.json
    metrics/metrics.csv              one row per epoch: train_*/val_* losses, agreement,
                                     accuracy, CKA per layer; train_grad_norm / _max and
                                     train_clip_fraction (every step, before clipping);
                                     diag_* gradient decomposition (diagnostics section)
    checkpoints/best.pt              weights at the best monitored epoch
    checkpoints/last.pt              full state, rewritten every epoch (resume)
    checkpoints/epochs/epochNNN.pt   every checkpoint.save_every_n_epochs
    logs/training.log

Usage (from DiCarlo_experiment/):
    python round3/train_student.py --config round3/configs/students/<run>.yaml
    python round3/train_student.py --config ... --resume     # continue from last.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# round3/ sits next to utils/: make "from utils..." (and round3 modules) importable.
ROUND3_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROUND3_DIR.parent
# round3/ must come BEFORE DiCarlo_experiment/: both contain a train_student.py
# (round 2's is one level up), and the round 3 one must win.
for _p in (ROUND3_DIR.parent, ROUND3_DIR):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from utils.activation_utils import capture_layer_responses, disable_response_storage
from utils.cache_data import CachedLoader, CachedSplit, CacheIndex, activation_key, read_batch, to_model_input
from utils.cka_loss import linear_cka

# Seed used to train the target checkpoint (from its DynVision config).
TARGET_TRAINING_SEED = 7000


# =============================================================================
# Configuration (defaults; every run's YAML overrides them)
# =============================================================================

@dataclass
class ExperimentInfo:
    # Free-text label only, e.g. "control", "aligned_IT". Written to the logs
    # and run_info.json. What the run DOES is decided by alignment.layers.
    role: str = "unnamed"
    description: str = ""


@dataclass
class CacheConfig:
    image_dir: Optional[str] = None    # prepare_images.py --out_dir
    target_dir: Optional[str] = None   # prepare_targets.py --out_dir


@dataclass
class DataConfig:
    # Split (utils/data_utils.build_or_load_split). Reuse split_file across
    # every run that is compared, or keep root/val_ratio/split_seed identical.
    root: Optional[str] = None
    val_ratio: float = 0.2
    split_seed: int = 2723
    split_file: Optional[str] = None

    batch_size: int = 48
    input_timesteps: int = 20
    # Train order = permutation(shuffle_seed + epoch). Same value in every
    # compared run -> same batches.
    shuffle_seed: int = 2723
    # DataLoader workers that read the cache files from disk.
    num_workers: int = 4


@dataclass
class StudentConfig:
    model_class: str = "dynvision.models:DyRCNNx4"
    config_path: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    use_current_defaults: bool = True
    initialize_lazy: bool = True
    probe_shape: Sequence[int] = (1, 20, 3, 224, 224)

    # Shared starting weights (make_student_init.py). Required.
    init_checkpoint: Optional[str] = None
    # Seed make_student_init.py uses to create them. Must differ from the
    # target's training seed (7000).
    init_seed: int = 2723


@dataclass
class AlignmentConfig:
    # Layers in the LOSS. Each entry: "IT" or {name: IT, timestep: -1, weight: 1.0}.
    # Empty list = control (CE only).
    layers: List[Any] = field(default_factory=list)
    # Layers whose CKA is only logged (no gradient). Same entry format;
    # weight is ignored. Layers in `layers` are always logged too.
    monitor_layers: List[Any] = field(default_factory=list)

    ce_weight: float = 1.0
    # CE averaged over timesteps [ce_first_timestep, T): the target's own
    # reaction-time masking (lossrt=8 ms, dt=2 ms -> first 4 steps ignored).
    ce_first_timestep: int = 4
    # Pseudo-label = argmax of the target's logits at this timestep.
    # -1 = the target's final prediction (decided: keep -1).
    pseudo_label_timestep: int = -1

    cka_eps: float = 1e-6          # floor for log(1 - CKA)
    warmup_epochs: int = 0         # CE-only epochs before the CKA terms switch on
    min_cka_batch_size: int = 16   # smaller batches: no CKA loss, not logged


@dataclass
class OptimizationConfig:
    epochs: int = 100
    # "adamw": decoupled weight decay (round 3). "adam": weight decay added to
    # the gradient (round 2).
    optimizer: str = "adamw"
    # Target's rule: lr = 2e-4 * sqrt(batch / 64)  (~1.41e-4 at batch 32).
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Sequence[float] = (0.9, 0.999)
    eps: float = 1e-8
    lr_group_factors: Dict[str, float] = field(
        default_factory=lambda: {"recurrence": 0.2, "feedback": 0.5}
    )
    accumulate_grad_batches: int = 1
    max_grad_norm: Optional[float] = None
    use_cosine_scheduler: bool = False
    cosine_t_max: Optional[int] = None


@dataclass
class RuntimeConfig:
    seed: int = 2723
    device: str = "cuda"
    precision: str = "bfloat16"     # float32 | bfloat16 | float16
    output_dir: str = "DiCarlo_experiment/outputs/student_run"
    log_every_n_steps: int = 20


@dataclass
class CheckpointConfig:
    # Any column of metrics.csv starting with "val_", e.g. val_total_loss,
    # val_ce_loss, val_cka_IT_t19 (mode "max").
    monitor: str = "val_total_loss"
    mode: str = "min"
    save_every_n_epochs: Optional[int] = None


@dataclass
class DiagnosticsConfig:
    # Once per epoch (after the train pass), on one FIXED train batch:
    # gradient norm of each loss term separately (CE, and every CKA term
    # with its weight), their cosine similarity, and the norm of their sum.
    # Written to metrics.csv as diag_* columns. Costs one extra
    # forward + one backward per term per epoch.
    gradient_terms: bool = True
    batch_size: Optional[int] = None   # None = data.batch_size
    seed: int = 0                      # which images form the fixed batch


@dataclass
class ExperimentConfig:
    experiment: ExperimentInfo = field(default_factory=ExperimentInfo)
    cache: CacheConfig = field(default_factory=CacheConfig)
    data: DataConfig = field(default_factory=DataConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


SECTIONS = tuple(ExperimentConfig.__dataclass_fields__)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise KeyError(f"Unknown config section(s) {sorted(unknown)}; expected {list(SECTIONS)}")
    cfg = ExperimentConfig()
    for section in SECTIONS:
        values = raw.get(section) or {}
        target = getattr(cfg, section)
        for key, value in values.items():
            if not hasattr(target, key):
                raise KeyError(f"Unknown config key '{section}.{key}'")
            setattr(target, key, value)
    return cfg


# =============================================================================
# Layer specs
# =============================================================================

@dataclass(frozen=True)
class LayerSpec:
    name: str
    timestep: int     # non-negative
    weight: float

    @property
    def key(self) -> str:
        return activation_key(self.name, self.timestep)


def parse_layer_specs(entries: Sequence[Any], n_timesteps: int, what: str) -> List[LayerSpec]:
    specs: List[LayerSpec] = []
    for entry in entries or []:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise ValueError(f"{what}: each entry must be a layer name or {{name, timestep, weight}}, got {entry!r}")
        extra = set(entry) - {"name", "timestep", "weight"}
        if extra:
            raise ValueError(f"{what}: unknown key(s) {sorted(extra)} in {dict(entry)}")
        t = int(entry.get("timestep", -1))
        t = t + n_timesteps if t < 0 else t
        if not 0 <= t < n_timesteps:
            raise ValueError(f"{what}: timestep {entry.get('timestep')} out of range for {n_timesteps} timesteps")
        specs.append(LayerSpec(str(entry["name"]), t, float(entry.get("weight", 1.0))))
    keys = [s.key for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{what}: duplicate layer/timestep entries {keys}")
    return specs


# =============================================================================
# General helpers
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_object(spec: str):
    import importlib
    module_name, object_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("alignment_training")  # same name data_utils logs to
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


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
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def autocast_context(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
    if dtype is None:
        raise ValueError("precision must be one of: float32, bfloat16, float16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def student_spec(cfg: ExperimentConfig) -> Dict[str, Any]:
    """The architecture-defining part of the student config (stored in the init file)."""
    s = cfg.student
    return {
        "model_class": s.model_class,
        "config_path": s.config_path,
        "overrides": dict(s.overrides),
        "use_current_defaults": bool(s.use_current_defaults),
        "initialize_lazy": bool(s.initialize_lazy),
        "probe_shape": [int(x) for x in s.probe_shape],
    }


def build_student(cfg: ExperimentConfig, device: torch.device, seed: int) -> torch.nn.Module:
    """Same construction path as before (create_aligned_control_pair); keeps one model."""
    from utils.student_model_utils import create_aligned_control_pair

    s = cfg.student
    model, twin = create_aligned_control_pair(
        model_cls=import_object(s.model_class),
        config_path=s.config_path,
        overrides=s.overrides,
        use_current_defaults=s.use_current_defaults,
        device=device,
        seed=seed,
        initialize_lazy=s.initialize_lazy,
        probe_shape=tuple(s.probe_shape),
    )
    del twin
    return model


# =============================================================================
# Loss / metric helpers
# =============================================================================

def temporal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, first_timestep: int) -> torch.Tensor:
    """CE averaged over timesteps [first_timestep, T), one label per sample."""
    n_t = logits.shape[1]
    start = first_timestep + n_t if first_timestep < 0 else first_timestep
    if not 0 <= start < n_t:
        raise ValueError(f"ce_first_timestep={first_timestep} out of range for {n_t} timesteps")
    supervised = logits[:, start:, :]
    labels = labels.unsqueeze(1).expand(-1, supervised.shape[1])
    return F.cross_entropy(supervised.reshape(-1, supervised.shape[-1]), labels.reshape(-1))


def build_param_groups(model, *, base_lr, lr_group_factors) -> Tuple[list, Dict[str, int]]:
    """Per-group LR, DynVision's naming rule (first matching key wins)."""
    named: Dict[str, list] = {name: [] for name in lr_group_factors}
    named["regular"] = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            group = next((g for g in lr_group_factors if g in param_name), "regular")
            named[group].append(param)
    groups, sizes = [], {}
    for name, params in named.items():
        if params:
            factor = 1.0 if name == "regular" else lr_group_factors[name]
            groups.append({"params": params, "lr": base_lr * factor, "name": name})
            sizes[name] = sum(p.numel() for p in params)
    return groups, sizes


class Meter:
    """Batch-size-weighted means; a None value is skipped for that key only."""

    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, values: Mapping[str, Optional[float]], n: int) -> None:
        for key, value in values.items():
            if value is None:
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def mean(self, key: str) -> float:
        return self.sums[key] / self.counts[key] if self.counts.get(key) else math.nan


# =============================================================================
# One epoch
# =============================================================================

def run_epoch(
    *,
    split: str,
    epoch: int,
    loader: CachedLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    align_specs: List[LayerSpec],
    monitor_specs: List[LayerSpec],
    cfg: ExperimentConfig,
    mean_std: Tuple[Sequence[float], Sequence[float]],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    is_train = split == "train"
    model.train(is_train)
    loader.set_epoch(epoch)

    a = cfg.alignment
    accum = max(1, int(cfg.optimization.accumulate_grad_batches))
    min_cka_batch = max(2, int(a.min_cka_batch_size))
    in_warmup = epoch < a.warmup_epochs
    n_batches = len(loader)

    # What the forward pass must capture: every layer name / timestep in use.
    all_specs = align_specs + monitor_specs
    cap_layers = tuple(dict.fromkeys(s.name for s in all_specs))
    cap_timesteps = tuple(sorted({s.timestep for s in all_specs}))

    meter = Meter()
    n_small = 0
    step_norms: List[float] = []   # total gradient norm before clipping, per optimizer step
    n_clipped = 0
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    grad_ctx = torch.enable_grad if is_train else torch.no_grad

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        pseudo = batch["pseudo"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
        x = to_model_input(images, mean=mean_std[0], std=mean_std[1], n_timesteps=cfg.data.input_timesteps)
        bsz = x.shape[0]

        use_cka = bsz >= min_cka_batch
        n_small += not use_cka

        with grad_ctx():
            with autocast_context(device, cfg.runtime.precision):
                if cap_layers:
                    logits, acts = capture_layer_responses(
                        model, x, layers=cap_layers, reset_state=True, timesteps=cap_timesteps,
                    )
                else:
                    if hasattr(model, "reset"):
                        model.reset()
                    logits, acts = model(x), {}
                ce = temporal_cross_entropy(logits, pseudo, a.ce_first_timestep)

            total = a.ce_weight * ce
            align_loss = None
            cka_values: Dict[str, Optional[float]] = {}

            for spec in align_specs:
                if not use_cka:
                    cka_values[f"cka_{spec.key}"] = None
                    continue
                student_repr = acts[spec.name][:, cap_timesteps.index(spec.timestep)]
                if in_warmup:
                    with torch.no_grad():
                        cka = linear_cka(student_repr.detach(), targets[spec.key])
                else:
                    cka = linear_cka(student_repr, targets[spec.key])
                    term = spec.weight * torch.log((1.0 - cka).clamp_min(a.cka_eps))
                    align_loss = term if align_loss is None else align_loss + term
                cka_values[f"cka_{spec.key}"] = cka.detach().item()

            if align_loss is not None:
                total = total + align_loss

            for spec in monitor_specs:
                if not use_cka:
                    cka_values[f"cka_{spec.key}"] = None
                    continue
                with torch.no_grad():
                    student_repr = acts[spec.name][:, cap_timesteps.index(spec.timestep)].detach()
                    cka_values[f"cka_{spec.key}"] = linear_cka(student_repr, targets[spec.key]).item()

        del acts, targets
        if is_train:
            (total / accum).backward()
            if (batch_idx + 1) % accum == 0 or batch_idx + 1 == n_batches:
                # clip_grad_norm_ returns the norm BEFORE clipping; with no
                # clipping configured, max_norm=inf only measures it.
                max_norm = cfg.optimization.max_grad_norm
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm if max_norm is not None else float("inf")).item()
                step_norms.append(norm)
                n_clipped += max_norm is not None and norm > max_norm
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "reset"):
            model.reset()  # drop the recurrent state kept after the forward

        with torch.no_grad():
            final = logits.detach()[:, a.pseudo_label_timestep].argmax(-1)
            values = {
                "total_loss": total.detach().item(),
                "ce_loss": ce.detach().item(),
                "align_loss": None if align_loss is None else align_loss.detach().item(),
                "target_agreement": (final == pseudo).float().mean().item(),
                "gt_accuracy": (final == gt).float().mean().item(),
                **cka_values,
            }
        meter.update(values, bsz)

        if is_train and (batch_idx + 1) % cfg.runtime.log_every_n_steps == 0:
            ckas = " ".join(f"{k[4:]}={v:.4f}" for k, v in cka_values.items() if v is not None)
            logger.info(
                "epoch=%d batch=%d/%d | total=%.5f ce=%.5f agree=%.4f | %s",
                epoch, batch_idx + 1, n_batches, values["total_loss"], values["ce_loss"],
                values["target_agreement"], ckas or "no CKA",
            )

    if n_small:
        logger.warning("%s epoch %d: %d batch(es) smaller than %d images -> no CKA for them.",
                       split, epoch, n_small, min_cka_batch)

    keys = ["total_loss", "ce_loss", "align_loss", "target_agreement", "gt_accuracy"]
    keys += [f"cka_{s.key}" for s in all_specs]
    out = {k: meter.mean(k) for k in keys}
    if is_train:
        out["grad_norm"] = float(np.mean(step_norms)) if step_norms else math.nan
        out["grad_norm_max"] = float(np.max(step_norms)) if step_norms else math.nan
        out["clip_fraction"] = n_clipped / len(step_norms) if step_norms else math.nan
    return out


# =============================================================================
# Gradient diagnostics (once per epoch, fixed batch)
# =============================================================================

def gradient_diagnostics(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    align_specs: List[LayerSpec],
    cfg: ExperimentConfig,
    mean_std: Tuple[Sequence[float], Sequence[float]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Gradient of each loss term on its own, w.r.t. all trainable parameters.

    Each term is taken WITH its weight, i.e. exactly what it contributes to
    the training gradient (ce_weight * CE, weight_l * log(1 - CKA_l)).
    Measured before clipping. Nothing is updated: torch.autograd.grad does
    not touch .grad or the optimizer.

    Columns:
      diag_grad_ce                 |grad of CE term|
      diag_grad_cka_<key>          |grad of that CKA term|
      diag_cos_ce_cka_<key>        cosine(CE grad, that CKA grad); < 0 = they pull apart
      diag_grad_cka_sum            |sum of all CKA-term grads|
      diag_ratio_cka_to_ce         diag_grad_cka_sum / diag_grad_ce
      diag_grad_total              |grad of the whole loss|
      diag_cka_<key>               CKA value on this batch
    """
    a = cfg.alignment
    params = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.train()

    x = to_model_input(batch["images"].to(device), mean=mean_std[0], std=mean_std[1],
                       n_timesteps=cfg.data.input_timesteps)
    pseudo = batch["pseudo"].to(device)
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    layers = tuple(dict.fromkeys(s.name for s in align_specs))
    timesteps = tuple(sorted({s.timestep for s in align_specs}))

    out: Dict[str, float] = {}
    with torch.enable_grad():
        with autocast_context(device, cfg.runtime.precision):
            if layers:
                logits, acts = capture_layer_responses(model, x, layers=layers, reset_state=True, timesteps=timesteps)
            else:
                if hasattr(model, "reset"):
                    model.reset()
                logits, acts = model(x), {}
            ce = temporal_cross_entropy(logits, pseudo, a.ce_first_timestep)

        terms: Dict[str, torch.Tensor] = {"ce": a.ce_weight * ce}
        for spec in align_specs:
            cka = linear_cka(acts[spec.name][:, timesteps.index(spec.timestep)], targets[spec.key])
            terms[f"cka_{spec.key}"] = spec.weight * torch.log((1.0 - cka).clamp_min(a.cka_eps))
            out[f"diag_cka_{spec.key}"] = cka.item()

        grads: Dict[str, torch.Tensor] = {}
        names = list(terms)
        for i, name in enumerate(names):
            g = torch.autograd.grad(terms[name], params, retain_graph=i < len(names) - 1, allow_unused=True)
            grads[name] = torch.cat([
                (gi if gi is not None else torch.zeros_like(p)).reshape(-1).float() for gi, p in zip(g, params)
            ])
    del acts, targets, logits, terms
    if hasattr(model, "reset"):
        model.reset()
    model.train(was_training)

    g_ce = grads["ce"]
    out["diag_grad_ce"] = g_ce.norm().item()
    cka_names = [n for n in names if n != "ce"]
    for name in cka_names:
        out[f"diag_grad_{name}"] = grads[name].norm().item()
        out[f"diag_cos_ce_{name}"] = F.cosine_similarity(g_ce, grads[name], dim=0).item()
    if cka_names:
        g_cka = torch.stack([grads[n] for n in cka_names]).sum(0)
        out["diag_grad_cka_sum"] = g_cka.norm().item()
        out["diag_ratio_cka_to_ce"] = out["diag_grad_cka_sum"] / max(out["diag_grad_ce"], 1e-12)
        out["diag_grad_total"] = (g_ce + g_cka).norm().item()
    else:
        out["diag_grad_total"] = out["diag_grad_ce"]
    return out


# =============================================================================
# Model card (one file with the full configuration of this student)
# =============================================================================

def build_model_card(cfg: ExperimentConfig, *, model, init_path: str, init_seed: int, index,
                     train_split, val_split, align_specs, monitor_specs, group_sizes,
                     config_path: Optional[Path]) -> Dict[str, Any]:
    from train_target import effective_model_values, layer_table, state_dict_shapes
    from utils.student_model_utils import resolve_student_kwargs

    s, a, o = cfg.student, cfg.alignment, cfg.optimization
    model_cls = import_object(s.model_class)
    kwargs = resolve_student_kwargs(model_cls=model_cls, config_path=s.config_path, overrides=s.overrides,
                                    use_current_defaults=s.use_current_defaults)
    target_dir = Path(cfg.cache.target_dir)
    target_card = target_dir / "target_model_card.yaml"
    try:
        import dynvision
        dynvision_dir = Path(dynvision.__file__).resolve().parents[1]
    except Exception:
        dynvision_dir = None
    loss = f"{a.ce_weight:g} * CE(pseudo-label) over timesteps {a.ce_first_timestep}..{cfg.data.input_timesteps - 1}"
    loss += "".join(f" + {sp.weight:g} * log(1 - CKA[{sp.key}])" for sp in align_specs)
    return {
        "model_card_version": 1,
        "name": Path(cfg.runtime.output_dir).name,
        "role": cfg.experiment.role,
        "description": cfg.experiment.description,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "training",
        "model": {
            "class": s.model_class,
            "kwargs": dict(kwargs),
            "kwargs_source": {"use_current_defaults (utils/student_model_utils.DEFAULT_STUDENT_KWARGS)":
                              bool(s.use_current_defaults), "config_path": s.config_path,
                              "overrides": dict(s.overrides)},
            "effective_values": effective_model_values(model, model_cls),
            "parameters_total": int(sum(p.numel() for p in model.parameters())),
            "parameters_by_module": layer_table(model),
            "state_dict_shapes": state_dict_shapes(model),
        },
        "init": {"path": str(Path(init_path).resolve()), "sha256_16": file_sha256_16(Path(init_path)),
                 "seed": int(init_seed)},
        "target": {
            "target_cache": str(target_dir.resolve()),
            "target_checkpoint": index.target_manifest["target"]["checkpoint"],
            "target_checkpoint_sha256_16": index.target_manifest["target"].get("checkpoint_sha256_16"),
            "target_model_card": str(target_card) if target_card.exists() else None,
        },
        "data": {
            **index.fingerprint(),
            "preprocessing": index.image_manifest.get("preprocessing"),
            "augmentation": "none",
            "split_file": cfg.data.split_file,
            "n_train": len(train_split), "n_val": len(val_split),
            "pseudo_label": f"argmax of the target's logits at timestep {a.pseudo_label_timestep}",
            "pseudo_label_equals_gt": {"train": train_split.pseudo_gt_agreement(),
                                       "val": val_split.pseudo_gt_agreement()},
            "input": "same image at every timestep",
            "batch_size": int(cfg.data.batch_size),
            "shuffle_seed": int(cfg.data.shuffle_seed),
        },
        "loss": {
            "formula": loss,
            "alignment_layers": [asdict(sp) | {"key": sp.key} for sp in align_specs],
            "monitored_only": [sp.key for sp in monitor_specs],
            "cka": "linear CKA (biased), per batch, float32",
            "cka_eps": float(a.cka_eps),
            "warmup_epochs": int(a.warmup_epochs),
            "min_cka_batch_size": int(a.min_cka_batch_size),
        },
        "training": {
            "epochs": int(o.epochs),
            "optimizer": o.optimizer,
            "learning_rate": float(o.learning_rate),
            "weight_decay": float(o.weight_decay),
            "weight_decay_type": "decoupled (AdamW)" if o.optimizer == "adamw" else "L2 added to gradient (Adam)",
            "betas": [float(b) for b in o.betas],
            "eps": float(o.eps),
            "lr_group_factors": dict(o.lr_group_factors),
            "lr_group_sizes": dict(group_sizes),
            "gradient_accumulation": int(o.accumulate_grad_batches),
            "max_grad_norm": o.max_grad_norm,
            "cosine_scheduler": bool(o.use_cosine_scheduler),
            "cosine_t_max_epochs": int(o.cosine_t_max or o.epochs) if o.use_cosine_scheduler else None,
            "precision": cfg.runtime.precision,
            "runtime_seed": int(cfg.runtime.seed),
        },
        "checkpoint_selection": {"evaluated_downstream": "last.pt",
                                 "best.pt": f"{cfg.checkpoint.monitor} ({cfg.checkpoint.mode})"},
        "environment": {
            "config_file": str(config_path.resolve()) if config_path else None,
            "repo_commit": git_commit(),
            "dynvision_path": str(dynvision_dir) if dynvision_dir else None,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


# =============================================================================
# Checkpoints / resume
# =============================================================================

def _is_better(value: float, best: Optional[float], mode: str) -> bool:
    if math.isnan(value):
        return False
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def save_weights(path: Path, model, epoch: int, metrics: Mapping[str, Any], cfg: ExperimentConfig) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": dict(metrics),
        "role": cfg.experiment.role,
        "config": asdict(cfg),
    }, path)


def save_last(path: Path, *, model, optimizer, scheduler, epoch, best_value, cfg) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_value": best_value,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "role": cfg.experiment.role,
        "config": asdict(cfg),
    }, tmp)
    tmp.replace(path)  # never leave a half-written last.pt behind


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_csv(path: Path, keep_through_epoch: int) -> None:
    """On resume: drop rows written after the last saved state."""
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    fields = list(rows[0])
    rows = [r for r in rows if int(r["epoch"]) <= keep_through_epoch]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main training function
# =============================================================================

def train_student(cfg: ExperimentConfig, *, resume: bool = False, config_path: Optional[Path] = None) -> Dict[str, Any]:
    # Checked before anything is written, so a failed start leaves no run folder.
    if not cfg.student.init_checkpoint or not Path(cfg.student.init_checkpoint).exists():
        raise FileNotFoundError(
            f"student.init_checkpoint={cfg.student.init_checkpoint!r} not found. Create it once with "
            f"make_student_init.py and point every compared run at the same file."
        )
    if cfg.cache.image_dir is None or cfg.cache.target_dir is None:
        raise ValueError("Set cache.image_dir and cache.target_dir.")

    out = Path(cfg.runtime.output_dir)
    dirs = {name: out / name for name in ("checkpoints", "metrics", "logs", "data")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(dirs["logs"] / "training.log")
    last_path = dirs["checkpoints"] / "last.pt"

    if last_path.exists() and not resume:
        raise FileExistsError(f"{last_path} exists. Pass --resume to continue, or use a new output_dir.")
    if resume and not last_path.exists():
        logger.warning("--resume given but %s does not exist: starting from scratch.", last_path)
        resume = False

    set_global_seed(cfg.runtime.seed)
    device = torch.device(cfg.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but it is not available.")
    torch.backends.cudnn.benchmark = True

    with (out / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    a = cfg.alignment
    T = cfg.data.input_timesteps
    align_specs = parse_layer_specs(a.layers, T, "alignment.layers")
    align_keys = {s.key for s in align_specs}
    monitor_specs = [s for s in parse_layer_specs(a.monitor_layers, T, "alignment.monitor_layers")
                     if s.key not in align_keys]

    logger.info("Role: %s | %s", cfg.experiment.role, cfg.experiment.description)
    logger.info("Loss: %.3g * CE%s", a.ce_weight,
                "".join(f" + {s.weight:g} * log(1 - CKA[{s.key}])" for s in align_specs) or "  (control: no CKA term)")
    logger.info("Monitored CKA only: %s", [s.key for s in monitor_specs] or "none")
    logger.info("Device %s | precision %s | seed %d", device, cfg.runtime.precision, cfg.runtime.seed)

    # ------------------------------------------------------------------
    # Data: split (same as before) -> rows of the caches.
    # ------------------------------------------------------------------
    from utils.data_utils import build_or_load_split

    index = CacheIndex(cfg.cache.image_dir, cfg.cache.target_dir)
    index.require([s.key for s in align_specs], "alignment.layers")
    missing_monitor = [s.key for s in monitor_specs if s.key not in index.target_manifest["activations"]]
    if missing_monitor:
        logger.warning("Not in the target cache, not monitored: %s", missing_monitor)
        monitor_specs = [s for s in monitor_specs if s.key not in missing_monitor]
    target_keys = [s.key for s in align_specs + monitor_specs]

    split = build_or_load_split(cfg)  # also saves <output_dir>/data/split.json
    train_split = CachedSplit(index, split["train"], name="train", target_keys=target_keys,
                              pseudo_label_timestep=a.pseudo_label_timestep)
    val_split = CachedSplit(index, split["val"], name="val", target_keys=target_keys,
                            pseudo_label_timestep=a.pseudo_label_timestep)
    for s in (train_split, val_split):
        logger.info("%s: %d images | target's final prediction == GT on %.4f", s.name, len(s), s.pseudo_gt_agreement())

    train_loader = CachedLoader(train_split, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True,
                                seed=cfg.data.shuffle_seed, num_workers=cfg.data.num_workers)
    val_loader = CachedLoader(val_split, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False,
                              num_workers=cfg.data.num_workers)
    mean_std = (index.mean, index.std)

    diag_batch = None
    if cfg.diagnostics.gradient_terms:
        n_diag = min(int(cfg.diagnostics.batch_size or cfg.data.batch_size), len(train_split))
        g = torch.Generator().manual_seed(int(cfg.diagnostics.seed))
        diag_positions = np.sort(torch.randperm(len(train_split), generator=g)[:n_diag].numpy())
        diag_batch = read_batch(train_split, diag_positions)
        logger.info("Gradient diagnostics: fixed batch of %d train images, once per epoch.", n_diag)
    logger.info("batch_size=%d | train batches/epoch=%d | val batches=%d",
                cfg.data.batch_size, len(train_loader), len(val_loader))

    # ------------------------------------------------------------------
    # Student from the shared init file.
    # ------------------------------------------------------------------
    init_path = cfg.student.init_checkpoint
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    if init.get("student_spec") != student_spec(cfg):
        raise RuntimeError(
            f"student.* in this config does not match the spec stored in {init_path}:\n"
            f"  init:   {init.get('student_spec')}\n  config: {student_spec(cfg)}"
        )
    model = build_student(cfg, device, seed=int(init["seed"]))
    model.load_state_dict(init["model_state_dict"], strict=True)
    disable_response_storage(model)

    groups, sizes = build_param_groups(model, base_lr=cfg.optimization.learning_rate,
                                       lr_group_factors=cfg.optimization.lr_group_factors)
    o = cfg.optimization
    if o.optimizer not in ("adamw", "adam"):
        raise ValueError(f"optimization.optimizer must be adamw or adam, got {o.optimizer!r}")
    optimizer_cls = torch.optim.AdamW if o.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(groups, lr=float(o.learning_rate), weight_decay=float(o.weight_decay),
                              betas=tuple(float(b) for b in o.betas), eps=float(o.eps))
    scheduler = None
    if cfg.optimization.use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.optimization.cosine_t_max or cfg.optimization.epochs)
    logger.info("%s lr=%g wd=%g | clip=%s | cosine=%s (T_max=%s) | groups %s", optimizer_cls.__name__,
                o.learning_rate, o.weight_decay, o.max_grad_norm, o.use_cosine_scheduler,
                o.cosine_t_max or o.epochs,
                {g["name"]: f"lr={g['lr']:g}, {sizes[g['name']]:,} params" for g in optimizer.param_groups})

    # ------------------------------------------------------------------
    # Resume?
    # ------------------------------------------------------------------
    start_epoch, best_value = 0, None
    metrics_path = dirs["metrics"] / "metrics.csv"
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best_value = state["epoch"] + 1, state["best_value"]
        truncate_csv(metrics_path, state["epoch"])
        logger.info("Resumed from %s: continuing at epoch %d (best %s=%s)",
                    last_path, start_epoch, cfg.checkpoint.monitor, best_value)
    else:
        info = {
            "role": cfg.experiment.role,
            "description": cfg.experiment.description,
            "git_commit": git_commit(),
            "loss_layers": [asdict(s) | {"key": s.key} for s in align_specs],
            "monitor_layers": [s.key for s in monitor_specs],
            "data": index.fingerprint(),
            "split_counts": {"train": len(train_split), "val": len(val_split)},
            "pseudo_label_vs_gt": {"train": train_split.pseudo_gt_agreement(),
                                   "val": val_split.pseudo_gt_agreement()},
            "student_init": {"path": str(Path(init_path).resolve()), "sha256_16": file_sha256_16(Path(init_path)),
                             "seed": int(init["seed"])},
        }
        with (out / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        from train_target import write_yaml
        card = build_model_card(cfg, model=model, init_path=init_path, init_seed=int(init["seed"]), index=index,
                                train_split=train_split, val_split=val_split, align_specs=align_specs,
                                monitor_specs=monitor_specs, group_sizes=sizes, config_path=config_path)
        write_yaml(out / "model_card.yaml", card)
        if cfg.runtime.seed == TARGET_TRAINING_SEED or int(init["seed"]) == TARGET_TRAINING_SEED:
            logger.warning("A seed equals the target's training seed (%d).", TARGET_TRAINING_SEED)

    # ------------------------------------------------------------------
    # Epochs.
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.optimization.epochs):
        t0 = time.time()
        logger.info("=" * 78)
        logger.info("Epoch %d / %d", epoch + 1, cfg.optimization.epochs)
        train_loader.set_epoch(epoch)
        order = train_loader.order_digest()

        common = dict(model=model, align_specs=align_specs, monitor_specs=monitor_specs, cfg=cfg,
                      mean_std=mean_std, device=device, logger=logger)
        train_m = run_epoch(split="train", epoch=epoch, loader=train_loader, optimizer=optimizer, **common)
        diag = {}
        if diag_batch is not None:
            diag = gradient_diagnostics(model=model, batch=diag_batch, align_specs=align_specs, cfg=cfg,
                                        mean_std=mean_std, device=device)
            logger.info("GRAD | step norm mean=%.3g max=%.3g clipped=%.0f%% | %s",
                        train_m["grad_norm"], train_m["grad_norm_max"], 100 * train_m["clip_fraction"],
                        " ".join(f"{k[5:]}={v:.3g}" for k, v in diag.items() if k.startswith("diag_grad")
                                 or k.startswith("diag_ratio") or k.startswith("diag_cos")))
        val_m = run_epoch(split="val", epoch=epoch, loader=val_loader, optimizer=None, **common)
        if scheduler is not None:
            scheduler.step()

        row: Dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row.update(diag)
        row["learning_rate"] = next(g["lr"] for g in optimizer.param_groups if g["name"] == "regular")
        row["train_order"] = order
        row["seconds"] = round(time.time() - t0, 1)
        append_csv(metrics_path, row)

        logger.info("VAL %s | total=%.5f ce=%.5f agree=%.4f gt_acc=%.4f | %s | %.0f s",
                    cfg.experiment.role, val_m["total_loss"], val_m["ce_loss"], val_m["target_agreement"],
                    val_m["gt_accuracy"],
                    " ".join(f"{k[4:]}={v:.4f}" for k, v in val_m.items() if k.startswith("cka_")) or "no CKA",
                    row["seconds"])

        every = cfg.checkpoint.save_every_n_epochs
        if every and (epoch + 1) % every == 0:
            (dirs["checkpoints"] / "epochs").mkdir(exist_ok=True)
            save_weights(dirs["checkpoints"] / "epochs" / f"epoch{epoch:03d}.pt", model, epoch, row, cfg)

        monitor = cfg.checkpoint.monitor
        if monitor not in row:
            raise KeyError(f"checkpoint.monitor '{monitor}' is not a metrics column: {sorted(row)}")
        # During warmup the CKA terms are off, so a loss that includes them is
        # not comparable with later epochs: no best selection then.
        if align_specs and epoch < a.warmup_epochs and monitor in ("val_total_loss", "val_align_loss"):
            logger.info("Warmup epoch: best-checkpoint selection skipped.")
        elif _is_better(float(row[monitor]), best_value, cfg.checkpoint.mode):
            best_value = float(row[monitor])
            save_weights(dirs["checkpoints"] / "best.pt", model, epoch, row, cfg)
            logger.info("New best.pt (%s=%.6f)", monitor, best_value)

        save_last(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                  epoch=epoch, best_value=best_value, cfg=cfg)

    summary = {"output_dir": str(out), "role": cfg.experiment.role,
               "monitor": cfg.checkpoint.monitor, "best_value": best_value}

    # Complete the model card with the final (last.pt) and best.pt numbers.
    from train_target import write_yaml
    card_path = out / "model_card.yaml"
    if card_path.exists():
        with card_path.open("r", encoding="utf-8") as f:
            card = yaml.safe_load(f)
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keep = [k for k in rows[-1] if k.startswith("val_") or k in ("train_ce_loss", "train_total_loss")]

        def numbers(r):
            return {k: float(r[k]) for k in keep if r.get(k) not in (None, "", "nan")}

        best_row = next((r for r in rows if best_value is not None
                         and math.isclose(float(r[cfg.checkpoint.monitor]), best_value, rel_tol=0, abs_tol=1e-12)), None)
        card["status"] = "trained"
        card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        card["checkpoints"] = {
            "evaluated_downstream": "last.pt",
            "last.pt": {"epoch": int(rows[-1]["epoch"]), **numbers(rows[-1])},
            "best.pt": ({"epoch": int(best_row["epoch"]), **numbers(best_row)} if best_row else None),
        }
        write_yaml(card_path, card)
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training complete. Outputs: %s", out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Run YAML (one student).")
    parser.add_argument("--resume", action="store_true", help="Continue from <output_dir>/checkpoints/last.pt.")
    args = parser.parse_args()
    train_student(load_experiment_config(args.config), resume=args.resume, config_path=Path(args.config))


if __name__ == "__main__":
    main()
