"""
Full training pipeline for DynVision target -> student neural alignment.

This file is intended to be the production/HPC training entry point.
A separate lightweight mock/sanity script should be used for local testing.

Core design
-----------
- Data loading is external and injected through a configurable factory.
- The SAME input tensor is passed to target, aligned student, and control.
- Target model is frozen and generates supervision online every batch:
    1) hard pseudo-labels from a configurable classifier timestep
    2) internal target representations/activations for CKA alignment
- Aligned and control models start from identical weights.
- Aligned model optimizes:
      CE + lambda_cka * log(1 - CKA)
- Control model optimizes:
      CE only
- CE uses the target's pseudo-label at every timestep from
  alignment.ce_first_timestep on (target's own reaction-time masking);
  optimizer recipe (Adam, weight decay, recurrence/feedback LR groups)
  follows the target's training. Data: utils/data_utils.py.
- Scalar histories, checkpoints, resolved config, logs, and optional
  per-epoch sample order are saved to disk.
- Visualization is intentionally NOT performed here.

Run later on Torch/HPC:
    python DiCarlo_experiment/train_alignment.py \
        --config DiCarlo_experiment/configs/alignment_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from utils.activation_utils import capture_layer_responses, disable_response_storage
from utils.cka_loss import linear_cka
from utils.pseudo_label_utils import extract_pseudo_labels
from utils.student_model_utils import create_aligned_control_pair
from utils.target_model_utils import load_target_model

# Seed used to train the target checkpoint (from its DynVision config).
TARGET_TRAINING_SEED = 7000


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataConfig:
    # External factory in "module:function" format.
    # Later example:
    #   utils.data_utils:build_dataloaders
    factory: str = "utils.data_utils:build_dataloaders"

    # Number of temporal presentations expected by the training pipeline.
    # If the dataloader returns [B,C,H,W], the SAME static image is expanded
    # over this many timesteps without any augmentation inside this script.
    input_timesteps: int = 20

    # If True and sample IDs are supplied by the dataloader, record the exact
    # per-epoch sample ordering/batch membership.
    save_epoch_sample_order: bool = True

    # ---- Used by utils/data_utils.py (see its docstring for the strategy) ----
    # Unpacked imagenette2-320 folder (contains train/ and val/), e.g.
    #   /scratch/<netid>/data/imagenette2-320
    root: Optional[str] = None

    # Fraction of the official train/ folder held out as OUR val set.
    val_ratio: float = 0.2
    split_seed: int = 2723

    # Reuse an existing split.json (e.g. from an earlier run) instead of
    # creating one. The split actually used is always saved to
    # <output_dir>/data/split.json.
    split_file: Optional[str] = None

    # 64: CKA quality vs memory trade-off (the target itself trained at 48).
    batch_size: int = 64
    num_workers: int = 4

    # Random augmentation on the TRAIN loader only (the target's own training
    # augmentation set). val/test are never augmented.
    train_augmentation: bool = True


@dataclass
class TargetConfig:
    checkpoint_path: Optional[str] = None
    pseudo_label_timestep: int = -1 # meaning we take the last timestep result


@dataclass
class StudentConfig:
    # Dynamic import in "module:ClassName" format.
    model_class: str = "dynvision.models:DyRCNNx4"

    # Optional DynVision YAML architecture source.
    config_path: Optional[str] = None

    # Explicit overrides applied on top of config/defaults.
    overrides: Dict[str, Any] = field(default_factory=dict)

    use_current_defaults: bool = True

    # Important for DynVision lazy skip/feedback adapters.
    initialize_lazy: bool = True
    probe_shape: Sequence[int] = (1, 20, 3, 224, 224)


@dataclass
class AlignmentConfig:
    layer: str = "IT" # align IT layer first
    timestep: int = -1 # the last timestep activations in IT layer will be aligned

    ce_weight: float = 1.0
    cka_weight: float = 1.0

    # CE is averaged over timesteps [ce_first_timestep, T), matching how the
    # target was trained: DynVision lossrt=8 ms with dt=2 ms masks the first
    # ceil(8/2) = 4 timesteps ("reaction time"), so CE covers steps 4..19.
    # Every supervised timestep uses the same pseudo-label (the target's
    # prediction at target.pseudo_label_timestep), just as the target used
    # one ground-truth label for every timestep. Set to -1 for last-step-only.
    ce_first_timestep: int = 4

    # Numerical floor for log(1 - CKA).
    cka_eps: float = 1e-6

    # Optional CE-only warmup. This is to match DiCarlo set up
    warmup_epochs: int = 0

    # Batches with fewer images than this get NO CKA loss and are excluded
    # from the logged batch CKA. CKA on a handful of images is strongly
    # inflated and its gradient is mostly noise (e.g. the small last batch of
    # an epoch when drop_last=False). DiCarlo used batches of 128.
    min_cka_batch_size: int = 16


@dataclass
class OptimizationConfig:
    epochs: int = 100

    # Base LR for batch_size 64. Target's rule: lr = 2e-4 * sqrt(batch / 64)
    # (2e-4 at 64, ~1.73e-4 at 48). Adjust by hand if batch_size changes.
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4  # as in the target's training

    # LR multipliers per parameter group, as in the target's training. A
    # parameter joins the first group whose key appears in its name
    # (DynVision's rule); everything else is "regular" (x1.0).
    lr_group_factors: Dict[str, float] = field(
        default_factory=lambda: {"recurrence": 0.2, "feedback": 0.5}
    )

    # Gradient accumulation helps with large recurrent models.
    accumulate_grad_batches: int = 1

    # Optional gradient clipping. None disables clipping.
    max_grad_norm: Optional[float] = None

    # Optional cosine scheduler.
    use_cosine_scheduler: bool = False
    cosine_t_max: Optional[int] = None


@dataclass
class RuntimeConfig:
    # Student init + loader shuffling/augmentation. Must differ from the
    # target's training seed (7000).
    seed: int = 2723
    device: str = "cuda"

    # "float32", "bfloat16", or "float16"
    precision: str = "bfloat16"

    output_dir: str = "DiCarlo_experiment/outputs/alignment_run"

    # Console/file logging frequency in batches.
    log_every_n_steps: int = 20


@dataclass
class CheckpointConfig:
    # aligned monitor choices:
    #   val_total_loss, val_ce_loss, val_batch_cka (mode "max")
    # Aligned best-checkpoint selection is skipped during warmup epochs,
    # because the CKA term is off then and val_total_loss is not comparable.
    aligned_monitor: str = "val_total_loss"
    aligned_mode: str = "min"

    # control monitor choices:
    #   val_ce_loss, val_batch_cka (mode "max")
    control_monitor: str = "val_ce_loss"
    control_mode: str = "min"

    save_last: bool = True

    # Save aligned + control weights every N epochs (None/0 = off), so the
    # evaluation pipeline can compare both models at the SAME training
    # points (DiCarlo-style trajectory). Files: checkpoints/epochs/
    # {aligned,control}_epoch{epoch:03d}.pt, with the 0-based epoch index
    # used in the metrics CSV. Weights + metrics only (no optimizer state),
    # to keep disk use down; these are for evaluation, not for resuming.
    save_every_n_epochs: Optional[int] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _update_dataclass(instance, values: Mapping[str, Any]):
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(
                f"Unknown config key '{key}' for {type(instance).__name__}"
            )
        setattr(instance, key, value)
    return instance


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()

    for section_name in (
        "data",
        "target",
        "student",
        "alignment",
        "optimization",
        "runtime",
        "checkpoint",
    ):
        if section_name in raw:
            _update_dataclass(
                getattr(cfg, section_name),
                raw[section_name] or {},
            )

    return cfg


# =============================================================================
# General helpers
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA, but torch.cuda.is_available() is False."
        )
    return torch.device(device_str)


def import_object(spec: str):
    """
    Import "module.path:object_name".
    """
    if ":" not in spec:
        raise ValueError(
            f"Expected import spec 'module:object', got: {spec}"
        )

    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)

    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_name}' has no object '{object_name}'."
        ) from exc


def setup_output_dirs(root: str | Path) -> Dict[str, Path]:
    root = Path(root)

    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "logs": root / "logs",
        "data": root / "data",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("alignment_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)

    return logger


def save_resolved_config(
    cfg: ExperimentConfig,
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(cfg),
            f,
            sort_keys=False,
        )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    metrics: Optional[Mapping[str, float]],
    config: ExperimentConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "metrics": dict(metrics or {}),
        "config": asdict(config),
    }
    torch.save(payload, path)


def append_metrics_csv(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    row = dict(row)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_sample_order_jsonl(
    path: Path,
    *,
    epoch: int,
    batch_idx: int,
    sample_ids: Sequence[Any],
) -> None:
    record = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "sample_ids": [
            x.item() if torch.is_tensor(x) and x.ndim == 0 else str(x)
            for x in sample_ids
        ],
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# =============================================================================
# Data interface
# =============================================================================

def build_dataloaders_from_factory(
    cfg: ExperimentConfig,
):
    """
    External dataset hook.

    The configured factory should return either:

        (train_loader, val_loader)

    or

        {
            "train": train_loader,
            "val": val_loader,
        }

    The future data_utils.py can therefore be developed independently of this
    training engine.
    """
    factory = import_object(cfg.data.factory)
    result = factory(cfg)

    if isinstance(result, Mapping):
        if "train" not in result or "val" not in result:
            raise KeyError(
                "Data factory mapping must contain 'train' and 'val'."
            )
        return result["train"], result["val"]

    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0], result[1]

    raise TypeError(
        "Data factory must return (train_loader, val_loader) "
        "or {'train': ..., 'val': ...}."
    )


def unpack_batch(batch):
    """
    Flexible batch parser.

    Supported forms
    ---------------
    dict:
        {
            "images": tensor,
            "labels": optional GT labels,
            "sample_ids": optional stable IDs,
        }

    tuple/list:
        (images,)
        (images, labels)
        (images, labels, sample_ids)

    Ground-truth labels are NOT used for CE supervision here.
    They are only retained for optional monitoring.
    """
    images = None
    gt_labels = None
    sample_ids = None

    if isinstance(batch, Mapping):
        images = batch.get("images", batch.get("image"))
        gt_labels = batch.get(
            "labels",
            batch.get("label", batch.get("targets")),
        )
        sample_ids = batch.get(
            "sample_ids",
            batch.get("sample_id", batch.get("indices")),
        )

    elif isinstance(batch, (tuple, list)):
        if len(batch) >= 1:
            images = batch[0]
        if len(batch) >= 2:
            gt_labels = batch[1]
        if len(batch) >= 3:
            sample_ids = batch[2]

    else:
        raise TypeError(
            f"Unsupported batch type: {type(batch).__name__}"
        )

    if images is None:
        raise ValueError("Could not find image tensor in batch.")

    return images, gt_labels, sample_ids


def ensure_temporal_input(
    images: torch.Tensor,
    *,
    n_timesteps: int,
) -> torch.Tensor:
    """
    Ensure input is in shape [B,T,C,H,W].

    IMPORTANT:
    This function performs NO random augmentation and NO image alteration.

    If input is [B,C,H,W], the exact same image tensor is presented at every
    timestep using an expanded view.

    If input is already [B,T,C,H,W], it is returned unchanged.
    """
    if images.ndim == 5:
        return images

    if images.ndim != 4:
        raise ValueError(
            "Expected images shaped [B,C,H,W] or [B,T,C,H,W], "
            f"got {tuple(images.shape)}."
        )

    return images.unsqueeze(1).expand(
        -1,
        n_timesteps,
        -1,
        -1,
        -1,
    )


# =============================================================================
# Precision helpers
# =============================================================================

def autocast_context(
    device: torch.device,
    precision: str,
):
    precision = precision.lower()

    if precision == "float32":
        return nullcontext()

    if device.type != "cuda":
        # Keep production behavior explicit rather than silently casting
        # unsupported CPU/MPS configurations.
        return nullcontext()

    if precision == "bfloat16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "float16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    raise ValueError(
        "precision must be one of: float32, bfloat16, float16"
    )


# =============================================================================
# Metrics
# =============================================================================

def classification_accuracy( # model prediction & ground truth label accuracy
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    timestep: int = -1,
) -> Optional[float]:
    if labels is None:
        return None

    labels = labels.to(logits.device)

    if logits.ndim == 3:
        logits = logits[:, timestep, :]

    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item() # turn logits into label


def target_agreement(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    timestep: int = -1,
) -> float:
    if logits.ndim == 3:
        logits = logits[:, timestep, :]

    preds = logits.argmax(dim=-1)
    return (preds == pseudo_labels).float().mean().item()


def select_timestep(
    tensor: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    """
    Select timestep from [B,T,...] -> [B,...].
    """
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected temporal tensor [B,T,...], got {tuple(tensor.shape)}"
        )
    return tensor[:, timestep, ...]


def temporal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    first_timestep: int,
) -> torch.Tensor:
    """
    CE averaged over timesteps [first_timestep, T), one label per sample.

    Same as DynVision's training loss with reaction-time masking: labels of
    the first `first_timestep` steps are ignored and CE is averaged over all
    remaining (sample, timestep) pairs. first_timestep=-1 -> last step only.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, C], got {tuple(logits.shape)}")

    n_timesteps = logits.shape[1]
    start = first_timestep + n_timesteps if first_timestep < 0 else first_timestep
    if not 0 <= start < n_timesteps:
        raise ValueError(
            f"ce_first_timestep={first_timestep} is out of range for "
            f"{n_timesteps} timesteps."
        )

    supervised = logits[:, start:, :]  # [B, T', C]
    n_supervised = supervised.shape[1]
    repeated_labels = labels.unsqueeze(1).expand(-1, n_supervised)  # [B, T']

    return F.cross_entropy(
        supervised.reshape(-1, supervised.shape[-1]),
        repeated_labels.reshape(-1),
    )


def build_param_groups(
    model: torch.nn.Module,
    *,
    base_lr: float,
    lr_group_factors: Mapping[str, float],
) -> Tuple[list, Dict[str, int]]:
    """
    Optimizer parameter groups with per-group LR (DynVision's naming rule).

    Returns the groups and a {group_name: n_parameters} summary for logging.
    """
    named: Dict[str, list] = {name: [] for name in lr_group_factors}
    named["regular"] = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = next((g for g in lr_group_factors if g in param_name), "regular")
        named[group].append(param)

    groups, summary = [], {}
    for group_name, params in named.items():
        if not params:
            continue
        factor = 1.0 if group_name == "regular" else lr_group_factors[group_name]
        groups.append({"params": params, "lr": base_lr * factor, "name": group_name})
        summary[group_name] = sum(p.numel() for p in params)

    return groups, summary


def alignment_loss_from_cka(
    cka_value: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Dapello-style neural loss:
        log(1 - CKA)

    We compute CKA once and reuse it for both logging and the loss.
    """
    return torch.log(
        (1.0 - cka_value).clamp_min(eps)
    )


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: Optional[float],
) -> None:
    """Optional gradient clipping, optimizer step, then clear gradients."""
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# Metrics reported for every batch.
_ALWAYS_METRICS = ("total_loss", "ce_loss", "target_agreement")
# Metrics that can be missing for a batch (small batch -> no CKA; control has
# no alignment loss; loader may not provide ground-truth labels). Each one
# keeps its own sample count so missing batches do not drag the mean down.
_OPTIONAL_METRICS = ("alignment_loss", "batch_cka", "gt_accuracy")


def _new_meter() -> Dict[str, float]:
    meter: Dict[str, float] = {"n": 0}
    for key in _ALWAYS_METRICS + _OPTIONAL_METRICS:
        meter[key] = 0.0
    for key in _OPTIONAL_METRICS:
        meter[f"{key}_n"] = 0
    return meter


def _update_meter(
    meter: Dict[str, float],
    values: Mapping[str, Optional[float]],
    batch_size: int,
) -> None:
    """Accumulate batch-size-weighted sums."""
    meter["n"] += batch_size
    for key in _ALWAYS_METRICS:
        meter[key] += float(values[key]) * batch_size
    for key in _OPTIONAL_METRICS:
        if values[key] is not None:
            meter[key] += float(values[key]) * batch_size
            meter[f"{key}_n"] += batch_size


def _finalize_meter(
    meter: Mapping[str, float],
    *,
    split: str,
) -> Dict[str, float]:
    if meter["n"] == 0:
        raise RuntimeError(f"{split} loader produced zero samples.")

    out = {key: meter[key] / meter["n"] for key in _ALWAYS_METRICS}
    for key in _OPTIONAL_METRICS:
        count = meter[f"{key}_n"]
        out[key] = meter[key] / count if count > 0 else math.nan
    return out


def _item_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    return None if value is None else value.detach().item()


def release_recurrent_state(model: torch.nn.Module) -> None:
    """
    Drop the hidden states a DynVision model keeps after a forward pass.

    Each layer holds its last few timesteps of activity (needed for the
    recurrence/feedback delays) until the model's NEXT forward resets them.
    With three models per batch that leftover state stacks up on the GPU
    (target + previous control are still resident during the aligned
    forward). Resetting right after use frees it. Results are unchanged:
    every forward starts with a reset anyway.
    """
    if hasattr(model, "reset"):
        model.reset()


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.5f}"


# =============================================================================
# One epoch
# =============================================================================

"""
拿一个 loader 跑完整个 epoch，同时让 target、aligned、control 都看完全相同的 
batch；target 负责提供 pseudo-label 和 representation supervision，aligned 
用 CE+CKA 训练，control 只用 CE 训练。
"""
def run_epoch(
    *,
    split: str,
    epoch: int,
    loader,
    target_model: torch.nn.Module,
    aligned_model: torch.nn.Module,
    control_model: torch.nn.Module,
    aligned_optimizer: Optional[torch.optim.Optimizer],
    control_optimizer: Optional[torch.optim.Optimizer],
    cfg: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    sample_order_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run one train or validation epoch for BOTH aligned and control models.

    Running them in the same batch loop guarantees that both conditions see
    exactly the same minibatches in exactly the same order.

    Memory: the aligned model does forward -> backward BEFORE the control
    model runs its forward, so only one student computation graph is alive
    at a time.

    About the reported CKA
    ----------------------
    `batch_cka` = CKA computed per batch, then averaged over batches
    (weighted by batch size). The per-batch CKA is the correct quantity for
    the training LOSS (DiCarlo does the same, batch size 128). But as a
    number it is only a training-time monitor: CKA is a property of a whole
    image set, so the batch average is NOT the dataset-level CKA. The
    aligned-vs-control comparison belongs in the separate evaluation
    pipeline, computed on the full held-out set.
    """
    is_train = split == "train"

    target_model.eval()
    aligned_model.train(is_train)
    control_model.train(is_train)

    n_batches = len(loader) if hasattr(loader, "__len__") else None

    if is_train:
        if aligned_optimizer is None or control_optimizer is None:
            raise ValueError("Training requires both optimizers.")
        # Checked BEFORE the loop: without __len__ the final partial
        # gradient-accumulation group could never be flushed.
        if n_batches is None:
            raise RuntimeError(
                "Training loader must implement __len__ so the final partial "
                "gradient-accumulation group can be detected."
            )
        aligned_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)

    accum_steps = max(1, int(cfg.optimization.accumulate_grad_batches))
    min_cka_batch = max(2, int(cfg.alignment.min_cka_batch_size))

    # Constant within an epoch, so computed once here.
    current_cka_weight = (
        0.0
        if epoch < cfg.alignment.warmup_epochs
        else cfg.alignment.cka_weight
    )

    layer = cfg.alignment.layer # IT
    label_t = cfg.target.pseudo_label_timestep # last timestep
    align_t = cfg.alignment.timestep # last time step

    meters = {"aligned": _new_meter(), "control": _new_meter()}
    n_small_batches = 0

    grad_context = torch.enable_grad if is_train else torch.no_grad

    # batch loop!
    for batch_idx, batch in enumerate(loader):
        images, gt_labels, sample_ids = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        if gt_labels is not None:
            gt_labels = gt_labels.to(device, non_blocking=True)

        # No augmentation happens here.
        # Create ONE temporal input tensor and reuse the SAME object for target,
        # aligned, and control.
        model_input = ensure_temporal_input(
            images,
            n_timesteps=cfg.data.input_timesteps, # 20
        )  # turn into DynVision input shape

        batch_size = model_input.shape[0]

        # Too few images for a meaningful CKA -> CE only for this batch.
        use_cka = batch_size >= min_cka_batch
        if not use_cka:
            n_small_batches += 1

        if (
            is_train
            and cfg.data.save_epoch_sample_order
            and sample_order_path is not None
            and sample_ids is not None
        ):
            if torch.is_tensor(sample_ids):
                ids_for_log = sample_ids.detach().cpu().tolist()
            else:
                ids_for_log = list(sample_ids)

            append_sample_order_jsonl(
                sample_order_path,
                epoch=epoch,
                batch_idx=batch_idx,
                sample_ids=ids_for_log,
            )

        should_step = is_train and (
            (batch_idx + 1) % accum_steps == 0 or batch_idx + 1 == n_batches
        )

        # -------------------------------------------------------------
        # Frozen target branch: online pseudo-label + representation.
        # -------------------------------------------------------------
        with torch.no_grad():  # no gradient update
            with autocast_context(device, cfg.runtime.precision):
                target_logits, target_acts = capture_layer_responses(
                    target_model,
                    model_input,
                    layers=(layer,),
                    reset_state=True,
                    timesteps=(align_t,),  # only what the CKA needs
                )

            # extract pseudo-labels from target logits
            pseudo_labels = extract_pseudo_labels(target_logits, timestep=label_t)

            # Activations were captured ONLY at align_t, so the time axis has
            # length 1 and index 0 IS align_t (same for aligned/control below).
            target_repr = target_acts[layer][:, 0]
            del target_acts, target_logits
        release_recurrent_state(target_model)

        # -------------------------------------------------------------
        # Aligned model: CE + cka_weight * log(1 - CKA).
        # -------------------------------------------------------------
        aligned_cka = None
        aligned_alignment_loss = None

        with grad_context():
            with autocast_context(device, cfg.runtime.precision):
                aligned_logits, aligned_acts = capture_layer_responses(
                    aligned_model,
                    model_input,
                    layers=(layer,),
                    reset_state=True,
                    timesteps=(align_t,),  # only what the CKA needs
                )

                # CE vs pseudo-labels over the supervised timesteps
                aligned_ce = temporal_cross_entropy(
                    aligned_logits,
                    pseudo_labels,
                    cfg.alignment.ce_first_timestep,
                )

            # extract the same layer activation at the same timestep
            aligned_repr = aligned_acts[layer][:, 0]
            del aligned_acts

            if use_cka:
                # linear_cka disables autocast internally -> float32 CKA.
                aligned_cka = linear_cka(aligned_repr, target_repr)
                aligned_alignment_loss = alignment_loss_from_cka(
                    aligned_cka,
                    cfg.alignment.cka_eps,
                )
                aligned_total = (
                    cfg.alignment.ce_weight * aligned_ce
                    + current_cka_weight * aligned_alignment_loss
                )
            else:
                aligned_total = cfg.alignment.ce_weight * aligned_ce

        if is_train:
            # Backward now, so this graph is freed before the control forward.
            (aligned_total / accum_steps).backward()
            if should_step:
                _optimizer_step(
                    aligned_model,
                    aligned_optimizer,
                    cfg.optimization.max_grad_norm,
                )
        del aligned_repr
        release_recurrent_state(aligned_model)

        # -------------------------------------------------------------
        # Control model: same pseudo-label CE, no alignment term.
        # -------------------------------------------------------------
        control_cka = None

        with grad_context():
            with autocast_context(device, cfg.runtime.precision):
                control_logits, control_acts = capture_layer_responses(
                    control_model,
                    model_input,
                    layers=(layer,),
                    reset_state=True,
                    timesteps=(align_t,),  # only what the CKA needs
                )

                control_ce = temporal_cross_entropy(
                    control_logits,
                    pseudo_labels,
                    cfg.alignment.ce_first_timestep,
                )

            # no CKA component for control model
            control_total = cfg.alignment.ce_weight * control_ce

            # Metric only. Detached so CKA can never affect control gradients.
            control_repr = control_acts[layer][:, 0].detach()
            del control_acts

        if use_cka:
            with torch.no_grad():
                control_cka = linear_cka(control_repr, target_repr)
        del control_repr, target_repr

        if is_train:
            (control_total / accum_steps).backward()
            if should_step:
                _optimizer_step(
                    control_model,
                    control_optimizer,
                    cfg.optimization.max_grad_norm,
                )
        release_recurrent_state(control_model)

        # -------------------------------------------------------------
        # Batch metrics.
        # -------------------------------------------------------------
        aligned_logits = aligned_logits.detach()
        control_logits = control_logits.detach()

        aligned_values = {
            "total_loss": aligned_total.detach().item(),
            "ce_loss": aligned_ce.detach().item(),
            "alignment_loss": _item_or_none(aligned_alignment_loss),
            "batch_cka": _item_or_none(aligned_cka),
            "target_agreement": target_agreement(
                aligned_logits, pseudo_labels, timestep=label_t
            ),
            "gt_accuracy": classification_accuracy(
                aligned_logits, gt_labels, timestep=label_t
            ),
        }

        control_values = {
            "total_loss": control_total.detach().item(),
            "ce_loss": control_ce.detach().item(),
            "alignment_loss": None,  # control has no alignment term
            "batch_cka": _item_or_none(control_cka),
            "target_agreement": target_agreement(
                control_logits, pseudo_labels, timestep=label_t
            ),
            "gt_accuracy": classification_accuracy(
                control_logits, gt_labels, timestep=label_t
            ),
        }

        _update_meter(meters["aligned"], aligned_values, batch_size)
        _update_meter(meters["control"], control_values, batch_size)

        if (
            is_train
            and (batch_idx + 1) % cfg.runtime.log_every_n_steps == 0
        ):
            logger.info(
                "epoch=%d batch=%d | "
                "aligned total=%.5f ce=%.5f align=%s batch_cka=%s agree=%.4f | "
                "control ce=%.5f batch_cka=%s agree=%.4f",
                epoch,
                batch_idx + 1,
                aligned_values["total_loss"],
                aligned_values["ce_loss"],
                _fmt(aligned_values["alignment_loss"]),
                _fmt(aligned_values["batch_cka"]),
                aligned_values["target_agreement"],
                control_values["ce_loss"],
                _fmt(control_values["batch_cka"]),
                control_values["target_agreement"],
            )

    if n_small_batches > 0:
        logger.warning(
            "%s epoch %d: %d batch(es) had fewer than %d images; CKA loss "
            "skipped and excluded from batch_cka. Consider drop_last=True "
            "for the train loader.",
            split,
            epoch,
            n_small_batches,
            min_cka_batch,
        )

    return {
        name: _finalize_meter(meter, split=split)
        for name, meter in meters.items()
    }


# =============================================================================
# Experiment orchestration
# =============================================================================

def _is_better(
    value: float,
    best: Optional[float],
    mode: str,
) -> bool:
    # A NaN monitor value (e.g. batch_cka when every batch was too small)
    # must never become the "best" value.
    if math.isnan(value):
        return False

    if best is None:
        return True

    if mode == "min":
        return value < best
    if mode == "max":
        return value > best

    raise ValueError("checkpoint mode must be 'min' or 'max'")

# put everything to work
def train_alignment_experiment(
    *,
    train_loader,
    val_loader,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Main reusable training function.

    The dataloaders are VARIABLES passed into this function. Dataset-specific
    logic therefore remains outside the training engine.
    """
    set_global_seed(cfg.runtime.seed)

    device = resolve_device(cfg.runtime.device)

    # Let cuDNN pick the fastest convolution algorithms for our fixed input
    # size (as in the target's training: trainer.benchmark=true).
    torch.backends.cudnn.benchmark = True
    dirs = setup_output_dirs(cfg.runtime.output_dir)

    logger = setup_logging(
        dirs["logs"] / "training.log"
    )

    save_resolved_config(
        cfg,
        dirs["root"] / "resolved_config.yaml",
    )

    logger.info("Device: %s", device)
    logger.info("Precision: %s", cfg.runtime.precision)
    logger.info("Alignment layer: %s", cfg.alignment.layer)
    logger.info("Alignment timestep: %s", cfg.alignment.timestep)
    logger.info("Pseudo-label timestep: %s", cfg.target.pseudo_label_timestep)
    logger.info("CE timesteps: from %s to the last", cfg.alignment.ce_first_timestep)
    logger.info("Seed: %d", cfg.runtime.seed)

    if cfg.runtime.seed == TARGET_TRAINING_SEED:
        logger.warning(
            "runtime.seed equals the target's training seed (%d). Use a "
            "different seed so the student cannot share the target's "
            "initialization.",
            TARGET_TRAINING_SEED,
        )

    # -----------------------------------------------------------------
    # Frozen target model.
    # -----------------------------------------------------------------
    target_model = load_target_model(
        checkpoint_path=cfg.target.checkpoint_path,
        device=device,
        freeze=True,
        verbose=True,
    )
    target_model.eval()
    disable_response_storage(target_model)

    # -----------------------------------------------------------------
    # Student model class + exactly matched aligned/control initialization.
    # -----------------------------------------------------------------
    student_model_cls = import_object(
        cfg.student.model_class
    )

    # two identical model set up for student + control models
    aligned_model, control_model = create_aligned_control_pair(
        model_cls=student_model_cls,
        config_path=cfg.student.config_path,
        overrides=cfg.student.overrides,
        use_current_defaults=cfg.student.use_current_defaults,
        device=device,
        seed=cfg.runtime.seed,
        initialize_lazy=cfg.student.initialize_lazy,
        probe_shape=tuple(cfg.student.probe_shape),
    )
    disable_response_storage(aligned_model)
    disable_response_storage(control_model)
    logger.info("DynVision response storage disabled for target, aligned, control.")

    # Save initial states immediately.
    save_checkpoint(
        dirs["checkpoints"] / "aligned_init.pt",
        model=aligned_model,
        optimizer=None,
        scheduler=None,
        epoch=-1,
        metrics=None,
        config=cfg,
    )
    save_checkpoint(
        dirs["checkpoints"] / "control_init.pt",
        model=control_model,
        optimizer=None,
        scheduler=None,
        epoch=-1,
        metrics=None,
        config=cfg,
    )

    # -----------------------------------------------------------------
    # Separate optimizers, identical hyperparameters and parameter groups
    # (recurrence / feedback weights get scaled LRs, as for the target).
    # -----------------------------------------------------------------
    optimizers = {}
    for name, model in (("aligned", aligned_model), ("control", control_model)):
        param_groups, group_sizes = build_param_groups(
            model,
            base_lr=cfg.optimization.learning_rate,
            lr_group_factors=cfg.optimization.lr_group_factors,
        )
        optimizers[name] = torch.optim.Adam(
            param_groups,
            weight_decay=cfg.optimization.weight_decay,
        )

    logger.info(
        "Optimizer: Adam, base lr=%g, weight_decay=%g, param groups %s",
        cfg.optimization.learning_rate,
        cfg.optimization.weight_decay,
        {
            g["name"]: f"lr={g['lr']:g}, {group_sizes[g['name']]:,} params"
            for g in optimizers["aligned"].param_groups
        },
    )

    aligned_optimizer = optimizers["aligned"]
    control_optimizer = optimizers["control"]

    aligned_scheduler = None
    control_scheduler = None

    if cfg.optimization.use_cosine_scheduler:
        t_max = (
            cfg.optimization.cosine_t_max
            or cfg.optimization.epochs
        )

        aligned_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            aligned_optimizer,
            T_max=t_max,
        )
        control_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            control_optimizer,
            T_max=t_max,
        )

    aligned_best_value = None
    control_best_value = None

    aligned_history_path = (
        dirs["metrics"] / "aligned_history.csv"
    )
    control_history_path = (
        dirs["metrics"] / "control_history.csv"
    )

    sample_order_path = (
        dirs["data"] / "train_batch_order.jsonl"
    )

    if sample_order_path.exists():
        sample_order_path.unlink()

    # -----------------------------------------------------------------
    # Epoch loop.
    # -----------------------------------------------------------------
    for epoch in range(cfg.optimization.epochs):
        logger.info("=" * 80)
        logger.info(
            "Epoch %d / %d",
            epoch + 1,
            cfg.optimization.epochs,
        )

        train_metrics = run_epoch(
            split="train",
            epoch=epoch,
            loader=train_loader,
            target_model=target_model,
            aligned_model=aligned_model,
            control_model=control_model,
            aligned_optimizer=aligned_optimizer,
            control_optimizer=control_optimizer,
            cfg=cfg,
            device=device,
            logger=logger,
            sample_order_path=sample_order_path,
        )

        val_metrics = run_epoch(
            split="val",
            epoch=epoch,
            loader=val_loader,
            target_model=target_model,
            aligned_model=aligned_model,
            control_model=control_model,
            aligned_optimizer=None, # no model update during validation
            control_optimizer=None,
            cfg=cfg,
            device=device,
            logger=logger,
            sample_order_path=None,
        )

        if aligned_scheduler is not None:
            aligned_scheduler.step()
            control_scheduler.step()

        aligned_lr = aligned_optimizer.param_groups[0]["lr"]
        control_lr = control_optimizer.param_groups[0]["lr"]

        aligned_row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["aligned"]["total_loss"],
            "train_ce_loss": train_metrics["aligned"]["ce_loss"],
            "train_alignment_loss": train_metrics["aligned"]["alignment_loss"],
            "train_batch_cka": train_metrics["aligned"]["batch_cka"],
            "train_target_agreement": train_metrics["aligned"]["target_agreement"],
            "train_gt_accuracy": train_metrics["aligned"]["gt_accuracy"],
            "val_total_loss": val_metrics["aligned"]["total_loss"],
            "val_ce_loss": val_metrics["aligned"]["ce_loss"],
            "val_alignment_loss": val_metrics["aligned"]["alignment_loss"],
            "val_batch_cka": val_metrics["aligned"]["batch_cka"],
            "val_target_agreement": val_metrics["aligned"]["target_agreement"],
            "val_gt_accuracy": val_metrics["aligned"]["gt_accuracy"],
            "learning_rate": aligned_lr,
        }

        control_row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["control"]["total_loss"],
            "train_ce_loss": train_metrics["control"]["ce_loss"],
            "train_alignment_loss": 0.0,
            "train_batch_cka": train_metrics["control"]["batch_cka"],
            "train_target_agreement": train_metrics["control"]["target_agreement"],
            "train_gt_accuracy": train_metrics["control"]["gt_accuracy"],
            "val_total_loss": val_metrics["control"]["total_loss"],
            "val_ce_loss": val_metrics["control"]["ce_loss"],
            "val_alignment_loss": 0.0,
            "val_batch_cka": val_metrics["control"]["batch_cka"],
            "val_target_agreement": val_metrics["control"]["target_agreement"],
            "val_gt_accuracy": val_metrics["control"]["gt_accuracy"],
            "learning_rate": control_lr,
        }

        append_metrics_csv(
            aligned_history_path,
            aligned_row,
        )
        append_metrics_csv(
            control_history_path,
            control_row,
        )

        logger.info(
            "VAL aligned: total=%.5f ce=%.5f align=%.5f "
            "batch_cka=%.5f agreement=%.4f",
            aligned_row["val_total_loss"],
            aligned_row["val_ce_loss"],
            aligned_row["val_alignment_loss"],
            aligned_row["val_batch_cka"],
            aligned_row["val_target_agreement"],
        )
        logger.info(
            "VAL control: ce=%.5f batch_cka=%.5f agreement=%.4f",
            control_row["val_ce_loss"],
            control_row["val_batch_cka"],
            control_row["val_target_agreement"],
        )

        # -------------------------------------------------------------
        # Periodic checkpoints (same epochs for aligned and control).
        # -------------------------------------------------------------
        every = cfg.checkpoint.save_every_n_epochs
        if every and (epoch + 1) % every == 0:
            epoch_dir = dirs["checkpoints"] / "epochs"
            epoch_dir.mkdir(exist_ok=True)

            for name, model, row in (
                ("aligned", aligned_model, aligned_row),
                ("control", control_model, control_row),
            ):
                save_checkpoint(
                    epoch_dir / f"{name}_epoch{epoch:03d}.pt",
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    epoch=epoch,
                    metrics=row,
                    config=cfg,
                )

            logger.info("Saved periodic checkpoints for epoch %d.", epoch)

        # -------------------------------------------------------------
        # Best checkpoints.
        # -------------------------------------------------------------
        aligned_monitor = cfg.checkpoint.aligned_monitor
        control_monitor = cfg.checkpoint.control_monitor

        if aligned_monitor not in aligned_row:
            raise KeyError(
                f"Unknown aligned monitor: {aligned_monitor}"
            )
        if control_monitor not in control_row:
            raise KeyError(
                f"Unknown control monitor: {control_monitor}"
            )

        aligned_value = float(
            aligned_row[aligned_monitor]
        )
        control_value = float(
            control_row[control_monitor]
        )

        # During warmup the CKA term is switched off, so val_total_loss is
        # on a different scale than after warmup (it jumps negative once
        # log(1 - CKA) is added). Comparing across that boundary would be
        # meaningless, so aligned "best" tracking starts after warmup.
        in_warmup = epoch < cfg.alignment.warmup_epochs

        if in_warmup:
            logger.info(
                "Warmup epoch: aligned best-checkpoint selection skipped."
            )
        elif _is_better(
            aligned_value,
            aligned_best_value,
            cfg.checkpoint.aligned_mode,
        ):
            aligned_best_value = aligned_value

            save_checkpoint(
                dirs["checkpoints"] / "aligned_best.pt",
                model=aligned_model,
                optimizer=aligned_optimizer,
                scheduler=aligned_scheduler,
                epoch=epoch,
                metrics=aligned_row,
                config=cfg,
            )

            logger.info(
                "Saved new aligned_best.pt (%s=%.6f)",
                aligned_monitor,
                aligned_value,
            )

        if _is_better(
            control_value,
            control_best_value,
            cfg.checkpoint.control_mode,
        ):
            control_best_value = control_value

            save_checkpoint(
                dirs["checkpoints"] / "control_best.pt",
                model=control_model,
                optimizer=control_optimizer,
                scheduler=control_scheduler,
                epoch=epoch,
                metrics=control_row,
                config=cfg,
            )

            logger.info(
                "Saved new control_best.pt (%s=%.6f)",
                control_monitor,
                control_value,
            )

    # -----------------------------------------------------------------
    # Last checkpoints.
    # -----------------------------------------------------------------
    if cfg.checkpoint.save_last:
        save_checkpoint(
            dirs["checkpoints"] / "aligned_last.pt",
            model=aligned_model,
            optimizer=aligned_optimizer,
            scheduler=aligned_scheduler,
            epoch=cfg.optimization.epochs - 1,
            metrics=None,
            config=cfg,
        )

        save_checkpoint(
            dirs["checkpoints"] / "control_last.pt",
            model=control_model,
            optimizer=control_optimizer,
            scheduler=control_scheduler,
            epoch=cfg.optimization.epochs - 1,
            metrics=None,
            config=cfg,
        )

    summary = {
        "output_dir": str(dirs["root"]),
        "aligned_best_monitor": cfg.checkpoint.aligned_monitor,
        "aligned_best_value": aligned_best_value,
        "control_best_monitor": cfg.checkpoint.control_monitor,
        "control_best_value": control_best_value,
    }

    with (dirs["root"] / "training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete.")
    logger.info("Outputs: %s", dirs["root"])

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to alignment YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)

    # Set up logging BEFORE building the loaders, so the data factory's
    # messages (split sizes, where split.json was saved) reach the log too.
    dirs = setup_output_dirs(cfg.runtime.output_dir)
    setup_logging(dirs["logs"] / "training.log")

    # Dataset-specific logic lives outside this file.
    train_loader, val_loader = build_dataloaders_from_factory(cfg)

    train_alignment_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()