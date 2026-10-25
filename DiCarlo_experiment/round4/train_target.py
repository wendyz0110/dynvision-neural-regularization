"""
Train the TARGET model from scratch on ground-truth labels (round 4).

Round 4 = round 3's script plus:
  * the initial weights are measured BEFORE training (same evaluate() as every
    epoch) and written as the first row of metrics.csv (epoch = -1,
    epochs_completed = 0); the initial weights are saved as checkpoints/init.pt.
  * optional data augmentation of the TRAINING batches (augmentation.transforms,
    torchvision.transforms.v2 strings in DynVision's format, e.g. its
    Imagenette preset). Validation, train-set evaluation, the target cache and
    everything downstream always use the un-augmented images.
  * metrics.csv has an `epochs_completed` column: 0 = initial weights,
    k = after k training epochs. Plot against this column.
  * load_trained_target(): rebuilds ANY trained target (any model class) from
    its model_card.yaml and loads its checkpoint strictly. prepare_targets.py
    and evaluate_alignment.py use it, so the target is no longer tied to the
    DyRCNNx4 architecture hard-coded in utils/target_model_utils.py.

Everything that defines the run is in one YAML (round4/configs/*.yaml):
the architecture (model class + every constructor argument), the data,
the loss and the training recipe. Nothing is taken from code defaults
except what DynVision itself fills in for arguments the YAML does not
give -- and those effective values are written to the model card.

Data: the uint8 image cache (prepare_images.py) + a split.json. Ground-truth
labels come from the image cache manifest. Augmentation: see AugmentationConfig. The target
cache (prepare_targets.py) is NOT needed: this script is what produces the
checkpoint that prepare_targets.py later reads.

Outputs in runtime.output_dir:
    model_card.yaml          THE single file describing this model: config,
                             effective architecture values, layers and
                             parameter counts, data, recipe, environment,
                             and (after training) which checkpoint is "the"
                             target and its metrics. model.class +
                             model.kwargs rebuild the network.
    resolved_config.yaml     the YAML as parsed (defaults filled in)
    data/split.json          copy of the split that was used
    metrics/metrics.csv      first row = initial weights (epochs_completed 0),
                             then one row per epoch (see train_epoch / evaluate)
    checkpoints/init.pt      plain state_dict, initial weights (before training)
    checkpoints/best.pt      plain state_dict, lowest val CE  <- the target
    checkpoints/last.pt      plain state_dict, final epoch
    checkpoints/epochs/epochNNN.pt   plain state_dict every N epochs
    checkpoints/resume.pt    full training state (optimizer, scheduler, RNG)
    logs/training.log

The plain state_dicts load with load_trained_target(<run dir>).

Usage (from DiCarlo_experiment/):
    python round4/train_target.py --config round4/configs/target_x8.yaml
    python round4/train_target.py --config ... --resume
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import logging
import math
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# round4/ sits next to utils/: make "from utils..." importable from anywhere.
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from utils.activation_utils import disable_response_storage
from utils.cache_data import _EpochBatchSampler, to_model_input
from utils.data_utils import load_split


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentInfo:
    name: str = "target"
    description: str = ""


@dataclass
class DataConfig:
    image_dir: Optional[str] = None     # prepare_images.py --out_dir
    split_file: Optional[str] = None    # split.json (train / val / test lists)
    batch_size: int = 32
    shuffle_seed: int = 7000            # train order = permutation(shuffle_seed + epoch)
    num_workers: int = 8


@dataclass
class AugmentationConfig:
    # Applied to TRAINING batches only, per image, on the cached (already
    # Resize(256) + CenterCrop(224)) uint8 images, on the GPU; the augmented
    # image is then shown at every timestep. Strings in DynVision's format:
    # a torchvision.transforms.v2 class with literal arguments, e.g.
    # "RandomRotation(10)". [] = no augmentation (round 3).
    # Random draws come from torch's global RNG (seeded by model.init_seed,
    # saved in resume.pt).
    transforms: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    # "module:Class". kwargs are passed to the constructor as they are; write
    # out every argument that matters -- the model card records the values
    # DynVision actually used, including its own defaults.
    model_class: str = "dynvision.models:DyRCNNx4"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    init_seed: int = 7000
    # Lazy skip/feedback adapters only exist after a forward pass over all
    # timesteps; a zero probe of this shape creates them before training.
    initialize_lazy: bool = True


@dataclass
class LossConfig:
    # CE vs ground truth, averaged over timesteps [ce_first_timestep, T).
    ce_first_timestep: int = 4


@dataclass
class OptimizationConfig:
    epochs: int = 200
    optimizer: str = "adamw"                 # adamw | adam
    learning_rate: float = 1.41e-4
    weight_decay: float = 0.01               # decoupled for adamw, L2 for adam
    betas: Sequence[float] = (0.9, 0.999)
    eps: float = 1e-8
    # Parameter name contains the key -> lr * factor (first match wins).
    lr_group_factors: Dict[str, float] = field(
        default_factory=lambda: {"recurrence": 0.2, "feedback": 0.5}
    )
    accumulate_grad_batches: int = 1
    max_grad_norm: Optional[float] = 500.0   # None = no clipping (norm still logged)
    scheduler: Optional[str] = "cosine"      # cosine | none
    cosine_t_max: Optional[int] = 200        # in epochs; None = epochs


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    precision: str = "bfloat16"              # float32 | bfloat16 | float16
    output_dir: Optional[str] = None
    log_every_n_steps: int = 20
    deterministic_cudnn: bool = False        # True = slower, bit-reproducible convs


@dataclass
class CheckpointConfig:
    # The checkpoint written to the model card as "the" target.
    select: str = "best"                     # best | last
    monitor: str = "val_ce_loss"             # any val_* column of metrics.csv
    mode: str = "min"
    save_every_n_epochs: Optional[int] = 10
    # Extra pass over the TRAIN images in eval mode every N epochs (and at
    # the last epoch): accuracy there vs val = how much the training set is
    # memorized. None = never.
    eval_train_every_n_epochs: Optional[int] = 10
    # Measure the initial weights before training (val + train in eval mode)
    # and write them as the first metrics.csv row (epochs_completed = 0).
    eval_init: bool = True
    # After training, check that the chosen checkpoint reloads with
    # load_trained_target() from the model card (strict) and reproduces the
    # logged val accuracy on a few batches. Failure is raised.
    verify_reload: bool = True


@dataclass
class TargetConfig:
    experiment: ExperimentInfo = field(default_factory=ExperimentInfo)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


SECTIONS = tuple(TargetConfig.__dataclass_fields__)


def load_config(path: str | Path) -> TargetConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise KeyError(f"Unknown config section(s) {sorted(unknown)}; expected {list(SECTIONS)}")
    cfg = TargetConfig()
    for section in SECTIONS:
        target = getattr(cfg, section)
        for key, value in (raw.get(section) or {}).items():
            if not hasattr(target, key):
                raise KeyError(f"Unknown config key '{section}.{key}'")
            setattr(target, key, value)
    return cfg


def validate_config(cfg: TargetConfig) -> None:
    if not cfg.data.image_dir or not cfg.data.split_file:
        raise ValueError("Set data.image_dir and data.split_file.")
    if not cfg.runtime.output_dir:
        raise ValueError("Set runtime.output_dir.")
    if not cfg.model.kwargs:
        raise ValueError("model.kwargs is empty: write the architecture out in the YAML.")
    if cfg.optimization.optimizer not in ("adamw", "adam"):
        raise ValueError("optimization.optimizer must be adamw or adam.")
    if cfg.optimization.scheduler not in (None, "none", "cosine"):
        raise ValueError("optimization.scheduler must be cosine or none.")
    if cfg.checkpoint.select not in ("best", "last"):
        raise ValueError("checkpoint.select must be best or last.")
    if cfg.checkpoint.mode not in ("min", "max"):
        raise ValueError("checkpoint.mode must be min or max.")


# =============================================================================
# Helpers
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_object(spec: str):
    module_name, object_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("target_training")
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


def git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def autocast_context(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
    if dtype is None:
        raise ValueError("precision must be one of: float32, bfloat16, float16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def plain(value: Any) -> Any:
    """Make a value YAML-safe (tuples -> lists, numpy/torch scalars -> python)."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if value is None or isinstance(value, bool):
        return value
    # Exact builtin types: safe_dump rejects subclasses (e.g. torch's TorchVersion).
    for t in (int, float, str):
        if isinstance(value, t):
            return t(value)
    return repr(value)


# =============================================================================
# Model
# =============================================================================

def build_model(cfg: TargetConfig) -> torch.nn.Module:
    """Seed -> construct from the YAML -> materialize lazy adapters (on CPU)."""
    m = cfg.model
    set_global_seed(int(m.init_seed))
    model = import_object(m.model_class)(**dict(m.kwargs))
    if m.initialize_lazy:
        n_t = int(m.kwargs.get("n_timesteps", 1))
        c, h, w = [int(v) for v in list(m.kwargs.get("input_dims", (1, 3, 224, 224)))[-3:]]
        probe = torch.zeros((1, n_t, c, h, w))
        model.eval()
        with torch.no_grad():
            try:
                model(probe, store_responses=False)
            except TypeError:
                model(probe)
        if hasattr(model, "reset"):
            model.reset()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


NOT_USED_BY_THIS_SCRIPT = {
    "learning_rate", "optimizer", "optimizer_kwargs", "optimizer_configs", "lr_parameter_groups",
    "scheduler", "scheduler_kwargs", "scheduler_configs", "criterion_params", "loss_reaction_time",
    "activity_loss_weight", "retain_graph", "target_dtype", "log_level", "log_every_n_steps",
    "store_responses", "store_responses_on_cpu", "store_train_responses", "store_val_responses",
    "store_test_responses", "store_train_records", "store_val_records", "store_test_records",
    "early_test_stop", "response_resolution",
}


def effective_model_values(model: torch.nn.Module, model_cls) -> Dict[str, Any]:
    """
    Every constructor argument of the class hierarchy, with the value the
    built model actually holds (attribute of the same name), else the
    default from the signature. This is what DynVision really used,
    including defaults the YAML did not set.
    """
    out: Dict[str, Any] = {}
    for klass in reversed(model_cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(inspect.unwrap(init)).parameters
        except (TypeError, ValueError):
            continue
        for name, p in params.items():
            if name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if hasattr(model, name) and not isinstance(getattr(model, name), (torch.nn.Module,)):
                out[name] = plain(getattr(model, name))
            elif p.default is not inspect.Parameter.empty:
                out[name] = plain(p.default)
    # DynVision models also carry settings for DynVision's own Lightning
    # training loop (optimizer, lr, scheduler, loss reaction time, logging,
    # response storage). This script does not use them -- the recipe is in
    # the model card's `training` section -- so they are left out here.
    for name in [k for k in out if k in NOT_USED_BY_THIS_SCRIPT]:
        del out[name]
    # Derived values that set the dynamics (exist on DynVision models).
    for name in ("history_length", "delay_index_skip", "delay_index_feedback", "delay_feedforward",
                 "n_timesteps", "layer_names", "layer_operations"):
        if hasattr(model, name):
            out[name] = plain(getattr(model, name))
    return dict(sorted(out.items()))


def layer_table(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Top-level modules with their class and own parameter count."""
    rows = []
    seen = set()
    for name, module in model.named_children():
        params = [p for p in module.parameters() if id(p) not in seen]
        seen.update(id(p) for p in params)
        rows.append({"name": name, "class": type(module).__name__,
                     "parameters": int(sum(p.numel() for p in params))})
    return rows


def state_dict_shapes(model: torch.nn.Module) -> Dict[str, List[int]]:
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def read_model_card(source: str | Path) -> Tuple[Dict[str, Any], Path]:
    """source = a target run dir, or a model card file (also the copy in a target cache)."""
    source = Path(source)
    card_path = source / "model_card.yaml" if source.is_dir() else source
    with card_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), card_path


def load_trained_target(source: str | Path, checkpoint_path: Optional[str | Path] = None, *,
                        device: str | torch.device = "cpu", freeze: bool = True) -> torch.nn.Module:
    """
    Rebuild a trained target from its model card and load its weights (strict).

    source: the target run dir (contains model_card.yaml) or a model card file.
    checkpoint_path: default = <run dir>/checkpoints/<card's target_checkpoint>.
    The architecture comes ONLY from the card (model.class + model.kwargs,
    lazy adapters created with the same zero probe as in training), so this
    works for any model class. If the card records the checkpoint's sha256,
    a different file is refused.
    """
    card, card_path = read_model_card(source)
    m = card["model"]
    if checkpoint_path is None:
        chosen = (card.get("checkpoints") or {}).get("target_checkpoint")
        if not chosen:
            raise ValueError(f"{card_path}: no checkpoints.target_checkpoint (target not finished?)")
        checkpoint_path = card_path.parent / "checkpoints" / chosen
    checkpoint_path = Path(checkpoint_path)
    recorded = (card.get("checkpoints") or {}).get(checkpoint_path.name, {})
    if isinstance(recorded, Mapping) and recorded.get("sha256_16"):
        if file_sha256_16(checkpoint_path) != recorded["sha256_16"]:
            raise RuntimeError(f"{checkpoint_path} differs from the {checkpoint_path.name} recorded in {card_path}.")
    cfg = TargetConfig()
    cfg.model.model_class = m["class"]
    cfg.model.kwargs = dict(m["kwargs"])
    cfg.model.init_seed = int(m.get("init_seed", 0))
    cfg.model.initialize_lazy = bool(m.get("lazy_adapters_initialized_with_zero_probe", True))
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    disable_response_storage(model)
    model.to(device)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def build_param_groups(model, *, base_lr: float, lr_group_factors: Mapping[str, float]):
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
            factor = 1.0 if name == "regular" else float(lr_group_factors[name])
            groups.append({"params": params, "lr": base_lr * factor, "name": name})
            sizes[name] = int(sum(p.numel() for p in params))
    return groups, sizes


def build_optimizer(model, o: OptimizationConfig):
    groups, sizes = build_param_groups(model, base_lr=float(o.learning_rate), lr_group_factors=o.lr_group_factors)
    cls = torch.optim.AdamW if o.optimizer == "adamw" else torch.optim.Adam
    optimizer = cls(groups, lr=float(o.learning_rate), weight_decay=float(o.weight_decay),
                    betas=tuple(float(b) for b in o.betas), eps=float(o.eps))
    scheduler = None
    if o.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(o.cosine_t_max or o.epochs))
    return optimizer, scheduler, sizes


# =============================================================================
# Data (image cache + ground-truth labels)
# =============================================================================

class ImageCacheSplit:
    """Rows of the image cache for one split; labels are the ground truth."""

    def __init__(self, image_dir: Path, items: Sequence[Sequence[Any]], name: str, manifest: Mapping[str, Any]):
        self.name = name
        self.array_file = Path(image_dir) / manifest["array_file"]
        row_of = {p: i for i, p in enumerate(manifest["paths"])}
        try:
            self.rows = np.asarray([row_of[str(p)] for p, _ in items], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"{name}: image {exc} from the split is not in the image cache.") from None
        labels = np.asarray([int(label) for _, label in items], dtype=np.int64)
        cache_labels = np.asarray(manifest["labels"], dtype=np.int64)[self.rows]
        if not np.array_equal(labels, cache_labels):
            raise RuntimeError(f"{name}: labels in the split disagree with the image cache.")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)


class _BatchReader(Dataset):
    def __init__(self, split: ImageCacheSplit):
        self.split = split
        self._images = None

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, positions: np.ndarray) -> Dict[str, torch.Tensor]:
        if self._images is None:
            self._images = np.load(self.split.array_file, mmap_mode="r")
        rows = self.split.rows[positions]
        return {
            "images": torch.from_numpy(np.asarray(self._images[rows])),
            "labels": torch.from_numpy(self.split.labels[positions]),
        }


class ImageLoader:
    def __init__(self, split: ImageCacheSplit, *, batch_size: int, shuffle: bool, drop_last: bool,
                 seed: int = 0, num_workers: int = 4):
        self.sampler = _EpochBatchSampler(len(split), batch_size, shuffle=shuffle, drop_last=drop_last, seed=seed)
        self.loader = DataLoader(_BatchReader(split), batch_size=None, sampler=self.sampler,
                                 num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                 persistent_workers=num_workers > 0)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        return iter(self.loader)


# =============================================================================
# Augmentation (training batches only)
# =============================================================================

def parse_transform(spec: str):
    """'RandomAffine(0, translate=(0.1, 0.1))' -> torchvision.transforms.v2.RandomAffine(0, translate=(0.1, 0.1)).
    Only a v2 class name with literal arguments is accepted."""
    from torchvision.transforms import v2
    node = ast.parse(spec.strip(), mode="eval").body
    if isinstance(node, ast.Name):
        name, args, kwargs = node.id, [], {}
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [ast.literal_eval(a) for a in node.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    else:
        raise ValueError(f"Cannot parse transform {spec!r}: expected Name or Name(literal args)")
    if not hasattr(v2, name):
        raise ValueError(f"{name!r} is not a torchvision.transforms.v2 transform")
    return getattr(v2, name)(*args, **kwargs)


class BatchAugment:
    """Applies the transform chain to every image of a uint8 batch separately
    (separate random draw per image). uint8 [B,3,H,W] in -> uint8 [B,3,H,W] out."""

    def __init__(self, specs: Sequence[str]):
        from torchvision.transforms import v2
        self.specs = list(specs)
        self.chain = v2.Compose([parse_transform(s) for s in self.specs])

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        out = torch.stack([self.chain(img) for img in images])
        if out.dtype != torch.uint8 or out.shape != images.shape:
            raise RuntimeError(f"augmentation changed dtype/shape: {images.dtype}{tuple(images.shape)} "
                               f"-> {out.dtype}{tuple(out.shape)}")
        return out


# =============================================================================
# Loss / one epoch
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


def forward(model, images, mean_std, n_timesteps, device, precision):
    x = to_model_input(images, mean=mean_std[0], std=mean_std[1], n_timesteps=n_timesteps)
    if hasattr(model, "reset"):
        model.reset()
    with autocast_context(device, precision):
        logits = model(x)
    return logits.float()


def train_epoch(*, epoch, loader, model, optimizer, cfg, mean_std, n_timesteps, device, logger,
                augment: Optional[BatchAugment] = None) -> Dict[str, float]:
    model.train()
    loader.set_epoch(epoch)
    accum = max(1, int(cfg.optimization.accumulate_grad_batches))
    max_norm = cfg.optimization.max_grad_norm
    n_batches = len(loader)
    sum_ce = sum_correct = n_seen = 0.0
    norms: List[float] = []
    n_clipped = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if augment is not None:
            images = augment(images)
        logits = forward(model, images, mean_std, n_timesteps, device, cfg.runtime.precision)
        ce = temporal_cross_entropy(logits, labels, cfg.loss.ce_first_timestep)
        (ce / accum).backward()
        if (batch_idx + 1) % accum == 0 or batch_idx + 1 == n_batches:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm if max_norm is not None else float("inf")).item()
            norms.append(norm)
            n_clipped += max_norm is not None and norm > max_norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "reset"):
            model.reset()

        bsz = labels.shape[0]
        with torch.no_grad():
            correct = (logits[:, -1].argmax(-1) == labels).sum().item()
        sum_ce += ce.item() * bsz
        sum_correct += correct
        n_seen += bsz
        if (batch_idx + 1) % cfg.runtime.log_every_n_steps == 0:
            logger.info("epoch=%d batch=%d/%d | ce=%.5f acc(t=last)=%.4f | grad_norm=%.3g",
                        epoch, batch_idx + 1, n_batches, ce.item(), correct / bsz, norms[-1] if norms else math.nan)

    return {
        "ce_loss": sum_ce / n_seen,
        "accuracy_running": sum_correct / n_seen,   # during training (weights change within the epoch; augmented images if augmentation is on)
        "grad_norm": float(np.mean(norms)) if norms else math.nan,
        "grad_norm_max": float(np.max(norms)) if norms else math.nan,
        "clip_fraction": n_clipped / len(norms) if norms else math.nan,
    }


@torch.no_grad()
def evaluate(*, loader, model, cfg, mean_std, n_timesteps, device) -> Dict[str, float]:
    """CE (same definition as training) and accuracy at every timestep, eval mode."""
    model.eval()
    sum_ce = n_seen = 0.0
    correct_t = None
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = forward(model, images, mean_std, n_timesteps, device, cfg.runtime.precision)
        if hasattr(model, "reset"):
            model.reset()
        bsz = labels.shape[0]
        sum_ce += temporal_cross_entropy(logits, labels, cfg.loss.ce_first_timestep).item() * bsz
        hits = (logits.argmax(-1) == labels.unsqueeze(1)).sum(0).double().cpu()   # [T]
        correct_t = hits if correct_t is None else correct_t + hits
        n_seen += bsz
    model.train()
    acc_t = (correct_t / n_seen).tolist()
    out = {"ce_loss": sum_ce / n_seen, "accuracy": acc_t[-1]}
    out.update({f"acc_t{t:02d}": a for t, a in enumerate(acc_t)})
    return out


# =============================================================================
# Checkpoints / metrics files
# =============================================================================

def save_state_dict(path: Path, model: torch.nn.Module) -> None:
    """Plain state_dict (same format as the DynVision trained.pt)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    tmp.replace(path)


def save_resume(path: Path, *, model, optimizer, scheduler, epoch, best, cfg) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best": best,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": asdict(cfg),
    }, tmp)
    tmp.replace(path)


def append_csv(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), restval="")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_csv(path: Path, keep_through_epoch: int) -> None:
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


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(plain(data), f, sort_keys=False, allow_unicode=True, width=120)
    tmp.replace(path)


# =============================================================================
# Main
# =============================================================================

def train_target(cfg: TargetConfig, *, config_path: Path, resume: bool = False) -> Dict[str, Any]:
    validate_config(cfg)
    out = Path(cfg.runtime.output_dir)
    dirs = {name: out / name for name in ("checkpoints", "metrics", "logs", "data")}
    resume_path = dirs["checkpoints"] / "resume.pt"
    if resume_path.exists() and not resume:
        raise FileExistsError(f"{resume_path} exists. Pass --resume to continue, or use a new output_dir.")
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(dirs["logs"] / "training.log")
    if resume and not resume_path.exists():
        logger.warning("--resume given but %s does not exist: starting from scratch.", resume_path)
        resume = False

    device = torch.device(cfg.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but it is not available.")
    torch.backends.cudnn.benchmark = not cfg.runtime.deterministic_cudnn
    torch.backends.cudnn.deterministic = bool(cfg.runtime.deterministic_cudnn)
    write_yaml(out / "resolved_config.yaml", asdict(cfg))

    # ---------------- data ----------------
    image_dir = Path(cfg.data.image_dir)
    with (image_dir / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("kind") != "image_cache":
        raise ValueError(f"{image_dir} is not an image cache.")
    split = load_split(Path(cfg.data.split_file))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = {p for p, _ in split[a]} & {p for p, _ in split[b]}
        if shared:
            raise RuntimeError(f"split overlap between {a} and {b}: {len(shared)} images")
    with (dirs["data"] / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    train_split = ImageCacheSplit(image_dir, split["train"], "train", manifest)
    val_split = ImageCacheSplit(image_dir, split["val"], "val", manifest)
    # The test images are never read here.
    mean_std = (tuple(manifest["normalize_mean"]), tuple(manifest["normalize_std"]))
    n_timesteps = int(cfg.model.kwargs.get("n_timesteps"))

    bs, nw = int(cfg.data.batch_size), int(cfg.data.num_workers)
    train_loader = ImageLoader(train_split, batch_size=bs, shuffle=True, drop_last=True,
                               seed=int(cfg.data.shuffle_seed), num_workers=nw)
    val_loader = ImageLoader(val_split, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw)
    train_eval_loader = ImageLoader(train_split, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw)
    logger.info("train %d images (%d batches/epoch) | val %d images | test never read",
                len(train_split), len(train_loader), len(val_split))
    augment = BatchAugment(cfg.augmentation.transforms) if cfg.augmentation.transforms else None
    logger.info("augmentation (training batches only): %s", augment.specs if augment else "none")

    # ---------------- model ----------------
    model_cls = import_object(cfg.model.model_class)
    model = build_model(cfg)
    disable_response_storage(model)
    model.to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer, scheduler, group_sizes = build_optimizer(model, cfg.optimization)
    o = cfg.optimization
    logger.info("Model %s | %d parameters | %s lr=%g wd=%g | clip=%s | scheduler=%s (T_max=%s)",
                cfg.model.model_class, n_params, o.optimizer, o.learning_rate, o.weight_decay,
                o.max_grad_norm, o.scheduler, o.cosine_t_max or o.epochs)
    logger.info("lr groups: %s", {g["name"]: f"lr={g['lr']:g}, {group_sizes[g['name']]:,} params"
                                  for g in optimizer.param_groups})

    # ---------------- model card (written now, completed at the end) ----------------
    card_path = out / "model_card.yaml"
    if resume and card_path.exists():
        with card_path.open("r", encoding="utf-8") as f:
            card = yaml.safe_load(f)
    else:
        try:
            import dynvision
            dynvision_dir = Path(dynvision.__file__).resolve().parents[1]
        except Exception:
            dynvision_dir = None
        card = {
            "model_card_version": 1,
            "name": cfg.experiment.name,
            "description": cfg.experiment.description,
            "role": "target (trained on ground-truth labels)",
            "created": datetime.now().isoformat(timespec="seconds"),
            "status": "training",
            "model": {
                "class": cfg.model.model_class,
                "kwargs": dict(cfg.model.kwargs),
                "init_seed": int(cfg.model.init_seed),
                "lazy_adapters_initialized_with_zero_probe": bool(cfg.model.initialize_lazy),
                "effective_values": effective_model_values(model, model_cls),
                "effective_values_note": ("every constructor argument with the value the built model holds "
                                          "(DynVision defaults included); DynVision's own Lightning-training "
                                          "fields are omitted because this script does not use them"),
                "parameters_total": n_params,
                "parameters_by_module": layer_table(model),
                "state_dict_shapes": state_dict_shapes(model),
            },
            "data": {
                "image_cache": str(image_dir.resolve()),
                "image_cache_paths_sha256_16": manifest.get("paths_sha256_16"),
                "preprocessing": manifest.get("preprocessing"),
                "normalize_mean": manifest["normalize_mean"],
                "normalize_std": manifest["normalize_std"],
                "augmentation": (list(cfg.augmentation.transforms) if cfg.augmentation.transforms else "none"),
                "augmentation_scope": "training batches only (val, train-set evaluation and target cache: un-augmented)",
                "split_file": str(Path(cfg.data.split_file).resolve()),
                "split_sha256_16": file_sha256_16(Path(cfg.data.split_file)),
                "n_train": len(train_split),
                "n_val": len(val_split),
                "n_test_never_used": len(split["test"]),
                "labels": "ground truth",
                "input": "same image at every timestep",
                "batch_size": bs,
                "shuffle_seed": int(cfg.data.shuffle_seed),
                "drop_last_train_batch": True,
            },
            "training": {
                "loss": f"cross-entropy vs ground truth, mean over timesteps {cfg.loss.ce_first_timestep}..{n_timesteps - 1}",
                "extra_loss_terms": "none",
                "epochs": int(o.epochs),
                "optimizer": o.optimizer,
                "learning_rate": float(o.learning_rate),
                "weight_decay": float(o.weight_decay),
                "weight_decay_type": "decoupled (AdamW)" if o.optimizer == "adamw" else "L2 added to gradient (Adam)",
                "betas": list(o.betas),
                "eps": float(o.eps),
                "lr_groups": {g["name"]: {"lr": g["lr"], "parameters": group_sizes[g["name"]]}
                              for g in optimizer.param_groups},
                "gradient_accumulation": int(o.accumulate_grad_batches),
                "effective_batch_size": bs * int(o.accumulate_grad_batches),
                "max_grad_norm": o.max_grad_norm,
                "scheduler": o.scheduler,
                "cosine_t_max_epochs": int(o.cosine_t_max or o.epochs) if o.scheduler == "cosine" else None,
                "precision": cfg.runtime.precision,
                "cudnn_deterministic": bool(cfg.runtime.deterministic_cudnn),
            },
            "checkpoint_selection": {
                "rule": f"{cfg.checkpoint.select} ({cfg.checkpoint.monitor}, {cfg.checkpoint.mode})"
                        if cfg.checkpoint.select == "best" else "last epoch",
            },
            "environment": {
                "config_file": str(Path(config_path).resolve()),
                "repo_commit": git_commit(EXPERIMENT_DIR),
                "dynvision_path": str(dynvision_dir) if dynvision_dir else None,
                "dynvision_commit": git_commit(dynvision_dir) if dynvision_dir else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "python": platform.python_version(),
                "host": platform.node(),
            },
        }
        write_yaml(card_path, card)

    # ---------------- resume ----------------
    start_epoch, best = 0, None   # best = {"epoch", "value"}
    metrics_path = dirs["metrics"] / "metrics.csv"
    if not resume and metrics_path.exists():
        # Left over from a run that died before its first resume.pt: keep it, but aside.
        stale = metrics_path.with_name(f"metrics_aborted_{datetime.now():%Y%m%d_%H%M%S}.csv")
        metrics_path.replace(stale)
        logger.warning("Old metrics.csv without resume.pt moved to %s", stale.name)
    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        truncate_csv(metrics_path, state["epoch"])
        logger.info("Resumed at epoch %d (best so far: %s)", start_epoch, best)

    fieldnames = (["epoch", "epochs_completed", "learning_rate"]
                  + [f"train_{k}" for k in ("ce_loss", "accuracy_running", "grad_norm", "grad_norm_max", "clip_fraction")]
                  + ["train_eval_ce_loss", "train_eval_accuracy"]
                  + ["val_ce_loss", "val_accuracy"] + [f"val_acc_t{t:02d}" for t in range(n_timesteps)]
                  + ["seconds"])
    ck = cfg.checkpoint
    epochs = int(o.epochs)
    common = dict(model=model, cfg=cfg, mean_std=mean_std, n_timesteps=n_timesteps, device=device)

    # ---------------- epochs_completed = 0: the initial weights ----------------
    if start_epoch == 0 and ck.eval_init:
        t0 = time.time()
        logger.info("=" * 78)
        logger.info("Initial weights (before training)")
        save_state_dict(dirs["checkpoints"] / "init.pt", model)
        val_m = evaluate(loader=val_loader, **common)
        train_eval = evaluate(loader=train_eval_loader, **common)
        row = {"epoch": -1, "epochs_completed": 0,
               "learning_rate": next(g["lr"] for g in optimizer.param_groups if g["name"] == "regular"),
               "train_eval_ce_loss": train_eval["ce_loss"], "train_eval_accuracy": train_eval["accuracy"]}
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row["seconds"] = round(time.time() - t0, 1)
        append_csv(metrics_path, row, fieldnames)
        logger.info("INIT val ce=%.5f acc(t=last)=%.4f | train_eval acc=%.4f | %.0f s",
                    val_m["ce_loss"], val_m["accuracy"], train_eval["accuracy"], row["seconds"])

    # ---------------- epochs ----------------
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logger.info("=" * 78)
        logger.info("Epoch %d / %d", epoch + 1, epochs)
        lr = next(g["lr"] for g in optimizer.param_groups if g["name"] == "regular")   # base lr
        train_m = train_epoch(epoch=epoch, loader=train_loader, optimizer=optimizer, logger=logger,
                              augment=augment, **common)
        val_m = evaluate(loader=val_loader, **common)
        last_epoch = epoch == epochs - 1
        train_eval = {}
        if ck.eval_train_every_n_epochs and ((epoch + 1) % int(ck.eval_train_every_n_epochs) == 0 or last_epoch):
            train_eval = evaluate(loader=train_eval_loader, **common)
        if scheduler is not None:
            scheduler.step()

        row: Dict[str, Any] = {"epoch": epoch, "epochs_completed": epoch + 1, "learning_rate": lr}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        if train_eval:
            row["train_eval_ce_loss"] = train_eval["ce_loss"]
            row["train_eval_accuracy"] = train_eval["accuracy"]
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row["seconds"] = round(time.time() - t0, 1)
        append_csv(metrics_path, row, fieldnames)

        logger.info("VAL ce=%.5f acc(t=last)=%.4f acc(t0)=%.4f | train ce=%.5f%s | grad %.3g (clipped %.0f%%) | %.0f s",
                    val_m["ce_loss"], val_m["accuracy"], val_m["acc_t00"], train_m["ce_loss"],
                    f" train_eval_acc={train_eval['accuracy']:.4f}" if train_eval else "",
                    train_m["grad_norm"], 100 * train_m["clip_fraction"], row["seconds"])

        if ck.save_every_n_epochs and (epoch + 1) % int(ck.save_every_n_epochs) == 0:
            (dirs["checkpoints"] / "epochs").mkdir(exist_ok=True)
            save_state_dict(dirs["checkpoints"] / "epochs" / f"epoch{epoch:03d}.pt", model)

        if ck.monitor not in row:
            raise KeyError(f"checkpoint.monitor '{ck.monitor}' is not a metrics column: {sorted(row)}")
        value = float(row[ck.monitor])
        better = (not math.isnan(value)) and (best is None or (value < best["value"] if ck.mode == "min"
                                                                else value > best["value"]))
        if better:
            best = {"epoch": epoch, "value": value}
            save_state_dict(dirs["checkpoints"] / "best.pt", model)
            logger.info("New best.pt (%s=%.6f)", ck.monitor, value)

        save_state_dict(dirs["checkpoints"] / "last.pt", model)
        save_resume(resume_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best=best, cfg=cfg)

    # ---------------- finish: complete the model card ----------------
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        rows = {int(r["epoch"]): r for r in csv.DictReader(f)}

    def summary_of(epoch: int) -> Dict[str, Any]:
        r = rows[epoch]
        keep = ["val_ce_loss", "val_accuracy", "train_ce_loss", "train_eval_accuracy"]
        out_d = {k: float(r[k]) for k in keep if r.get(k) not in (None, "")}
        out_d["val_accuracy_by_timestep"] = [round(float(r[f"val_acc_t{t:02d}"]), 4) for t in range(n_timesteps)]
        return out_d

    last_e = epochs - 1
    chosen = "best.pt" if ck.select == "best" else "last.pt"
    card["status"] = "trained"
    card["finished"] = datetime.now().isoformat(timespec="seconds")
    card["checkpoints"] = {
        "target_checkpoint": chosen,
        "best.pt": {"epoch": best["epoch"], "epochs_completed": best["epoch"] + 1, ck.monitor: best["value"],
                    "sha256_16": file_sha256_16(dirs["checkpoints"] / "best.pt"), **summary_of(best["epoch"])},
        "last.pt": {"epoch": last_e, "epochs_completed": last_e + 1, "sha256_16": file_sha256_16(dirs["checkpoints"] / "last.pt"),
                    **summary_of(last_e)},
    }
    if (dirs["checkpoints"] / "init.pt").exists():
        card["checkpoints"]["init.pt"] = {"epoch": -1, "epochs_completed": 0, "note": "initial weights, before training",
                                          "sha256_16": file_sha256_16(dirs["checkpoints"] / "init.pt"),
                                          **(summary_of(-1) if -1 in rows else {})}
    write_yaml(card_path, card)
    if ck.verify_reload:
        reloaded = load_trained_target(out, device=device)
        n_check = min(len(val_split), 4 * bs)
        sub = ImageCacheSplit(image_dir, split["val"][:n_check], "val_check", manifest)
        check = evaluate(loader=ImageLoader(sub, batch_size=bs, shuffle=False, drop_last=False, num_workers=0),
                         model=reloaded, cfg=cfg, mean_std=mean_std, n_timesteps=n_timesteps, device=device)
        model.load_state_dict(torch.load(dirs["checkpoints"] / chosen, map_location="cpu"))
        ref = evaluate(loader=ImageLoader(sub, batch_size=bs, shuffle=False, drop_last=False, num_workers=0),
                       **common)
        if abs(check["ce_loss"] - ref["ce_loss"]) > 1e-3:
            raise RuntimeError(f"Reloaded target differs: val CE {check['ce_loss']} vs {ref['ce_loss']}")
        card["checkpoints"]["reload_check"] = {"loader": "round4/train_target.py:load_trained_target",
                                               "images": n_check, "val_ce_reloaded": check["ce_loss"],
                                               "val_ce_in_training_process": ref["ce_loss"], "ok": True}
        write_yaml(card_path, card)
        logger.info("[OK] %s reloads with load_trained_target (val CE %.5f on %d images).",
                    chosen, check["ce_loss"], n_check)
    logger.info("Training complete. Target checkpoint: %s (epoch %d). Model card: %s",
                dirs["checkpoints"] / chosen, best["epoch"] if ck.select == "best" else last_e, card_path)
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Target YAML.")
    parser.add_argument("--resume", action="store_true", help="Continue from <output_dir>/checkpoints/resume.pt.")
    args = parser.parse_args()
    train_target(load_config(args.config), config_path=Path(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
