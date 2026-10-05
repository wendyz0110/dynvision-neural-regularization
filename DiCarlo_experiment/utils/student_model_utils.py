
"""
Flexible student/control model construction utilitie.

Key principles
--------------
1. The student is NOT permanently hard-coded to equal the current target.
2. Student architecture can come from explicit kwargs, a YAML config,
   or a custom model class.
3. Aligned and control models are separately instantiated model objects,
   but are loaded from the exact same initial state_dict.
4. We intentionally do NOT use copy.deepcopy(model), because DynVision
   models contain thread locks (RLock) inside their storage machinery,
   which are not pickle/deepcopy compatible.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

import torch
import yaml

from dynvision.models import DyRCNNx4


# ---------------------------------------------------------------------
# Current experiment defaults.
# These are defaults only, not permanent architectural constraints.
# ---------------------------------------------------------------------

DEFAULT_STUDENT_KWARGS: Dict[str, Any] = {
    "n_classes": 10,
    "input_dims": (1, 3, 224, 224),
    "n_timesteps": 20,
    "dt": 2,
    "tau": 9,
    "t_feedforward": 0,
    "t_recurrence": 6,
    "t_feedback": 12,
    "t_skip": 0,
    "recurrence_type": "pointdepthwise",
    "recurrence_target": "output",
    "skip": True,
    "feedback": "add",
}


_CONFIG_ALIASES = {
    "n_classes": ("n_classes", "model.n_classes"),
    "input_dims": ("input_dims", "model.input_dims"),
    "n_timesteps": ("n_timesteps", "model.n_timesteps", "tsteps", "model_args.tsteps"),
    "dt": ("dt", "model.dt", "model_args.dt"),
    "tau": ("tau", "model.tau", "model_args.tau"),
    "t_feedforward": ("t_feedforward", "model.t_feedforward", "tff", "model_args.tff"),
    "t_recurrence": ("t_recurrence", "model.t_recurrence", "trc", "model_args.trc"),
    "t_feedback": ("t_feedback", "model.t_feedback", "tfb", "model_args.tfb"),
    "t_skip": ("t_skip", "model.t_skip", "tsk", "model_args.tsk"),
    "recurrence_type": (
        "recurrence_type",
        "model.recurrence_type",
        "rctype",
        "model_args.rctype",
    ),
    "recurrence_target": (
        "recurrence_target",
        "model.recurrence_target",
        "rctarget",
        "model_args.rctarget",
    ),
    "skip": ("skip", "model.skip", "model_args.skip"),
    "feedback": ("feedback", "model.feedback", "model_args.feedback"),
}


def _flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """Flatten nested dictionaries into dotted keys."""
    out: Dict[str, Any] = {}

    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        out[full_key] = value

        if isinstance(value, Mapping):
            out.update(_flatten_mapping(value, full_key))

    return out


def _normalize_bool_like(value: Any) -> Any:
    """Convert common string booleans to Python bools."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def load_model_kwargs_from_yaml(
    config_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Extract architecture/temporal constructor kwargs from a DynVision YAML file.

    Training-only fields such as optimizer, batch size, loss settings, and
    checkpoint paths are intentionally ignored.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found:\n{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, Mapping):
        raise ValueError("Expected YAML config to contain a mapping/dictionary.")

    flat = _flatten_mapping(raw)
    extracted: Dict[str, Any] = {}

    for constructor_name, aliases in _CONFIG_ALIASES.items():
        for key in aliases:
            if key in flat and not isinstance(flat[key], Mapping):
                extracted[constructor_name] = _normalize_bool_like(flat[key])
                break

    if "input_dims" in extracted:
        extracted["input_dims"] = tuple(extracted["input_dims"])

    return extracted


def _filter_kwargs_for_constructor(
    model_cls: Type[torch.nn.Module],
    kwargs: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Validate kwargs against the selected model constructor."""
    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )

    if accepts_var_kwargs:
        return dict(kwargs)

    valid_names = {
        name
        for name, p in parameters.items()
        if name != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    unknown = sorted(set(kwargs) - valid_names)

    if unknown and strict:
        raise TypeError(
            f"{model_cls.__name__} does not accept these constructor arguments: "
            f"{unknown}\n"
            "For a structurally different student, use supported constructor "
            "kwargs or pass a custom model class."
        )

    return {k: v for k, v in kwargs.items() if k in valid_names}


def resolve_student_kwargs(
    *,
    model_cls: Type[torch.nn.Module] = DyRCNNx4,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    use_current_defaults: bool = True,
    strict_constructor_args: bool = True,
) -> Dict[str, Any]:
    """
    Resolve the final constructor kwargs without instantiating a model.

    Precedence, lowest -> highest:
        current defaults
        YAML config
        model_kwargs
        overrides
    """
    kwargs: Dict[str, Any] = {}

    if use_current_defaults:
        kwargs.update(DEFAULT_STUDENT_KWARGS)

    if config_path is not None:
        kwargs.update(load_model_kwargs_from_yaml(config_path))

    if model_kwargs is not None:
        kwargs.update(dict(model_kwargs))

    if overrides is not None:
        kwargs.update(dict(overrides))

    return _filter_kwargs_for_constructor(
        model_cls,
        kwargs,
        strict=strict_constructor_args,
    )


def build_student_model(
    *,
    model_cls: Type[torch.nn.Module] = DyRCNNx4,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    use_current_defaults: bool = True,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    strict_constructor_args: bool = True,
) -> torch.nn.Module:
    """Build a fresh, trainable student model."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    kwargs = resolve_student_kwargs(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        config_path=config_path,
        overrides=overrides,
        use_current_defaults=use_current_defaults,
        strict_constructor_args=strict_constructor_args,
    )

    model = model_cls(**kwargs)
    model.to(torch.device(device))
    model.train()

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    return model


def initialize_lazy_connections(
    model: torch.nn.Module,
    *,
    probe_shape: Tuple[int, int, int, int, int],
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Materialize lazy DynVision connections with a dummy forward.

    probe_shape is [B, T, C, H, W].
    """
    device = torch.device(device)
    model.to(device)

    probe = torch.zeros(probe_shape, dtype=torch.float32, device=device)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        try:
            _ = model(probe, store_responses=False)
        except TypeError:
            _ = model(probe)

    if hasattr(model, "reset"):
        model.reset()

    model.train(was_training)


def _clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Make an independent tensor copy of a state_dict.

    This is safe even when the model itself contains non-copyable objects such
    as thread locks.
    """
    return {
        key: value.detach().clone()
        for key, value in state_dict.items()
    }


def create_aligned_control_pair(
    *,
    model_cls: Type[torch.nn.Module] = DyRCNNx4,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    use_current_defaults: bool = True,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    initialize_lazy: bool = False,
    probe_shape: Optional[Tuple[int, int, int, int, int]] = None,
    strict_constructor_args: bool = True,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create aligned and control models with exactly identical initial weights.

    Implementation:
        1. Resolve one architecture specification.
        2. Build ONE base model.
        3. Optionally materialize all lazy connections on the base model.
        4. Snapshot its complete state_dict.
        5. Build two fresh model objects from the same architecture spec.
        6. Materialize the same lazy modules on both when requested.
        7. Strictly load the same base state_dict into both.

    This avoids copy.deepcopy(), which fails for DynVision because its storage
    machinery contains non-pickleable thread.RLock objects.
    """
    device = torch.device(device)

    resolved_kwargs = resolve_student_kwargs(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        config_path=config_path,
        overrides=overrides,
        use_current_defaults=use_current_defaults,
        strict_constructor_args=strict_constructor_args,
    )

    # ------------------------------------------------------------
    # Build one reference initialization.
    # ------------------------------------------------------------
    base_model = build_student_model(
        model_cls=model_cls,
        model_kwargs=resolved_kwargs,
        use_current_defaults=False,
        device=device,
        seed=seed,
        strict_constructor_args=strict_constructor_args,
    )

    if initialize_lazy:
        if probe_shape is None:
            raise ValueError(
                "probe_shape is required when initialize_lazy=True."
            )
        initialize_lazy_connections(
            base_model,
            probe_shape=probe_shape,
            device=device,
        )

    initial_state = _clone_state_dict(base_model.state_dict())

    # ------------------------------------------------------------
    # Build two independent model objects.
    # ------------------------------------------------------------
    aligned_model = build_student_model(
        model_cls=model_cls,
        model_kwargs=resolved_kwargs,
        use_current_defaults=False,
        device=device,
        seed=seed,
        strict_constructor_args=strict_constructor_args,
    )

    control_model = build_student_model(
        model_cls=model_cls,
        model_kwargs=resolved_kwargs,
        use_current_defaults=False,
        device=device,
        seed=seed,
        strict_constructor_args=strict_constructor_args,
    )

    # Lazy modules must exist before strict state loading if they were part of
    # the base snapshot.
    if initialize_lazy:
        initialize_lazy_connections(
            aligned_model,
            probe_shape=probe_shape,
            device=device,
        )
        initialize_lazy_connections(
            control_model,
            probe_shape=probe_shape,
            device=device,
        )

    aligned_model.load_state_dict(initial_state, strict=True)
    control_model.load_state_dict(initial_state, strict=True)

    aligned_model.train()
    control_model.train()

    for model in (aligned_model, control_model):
        if hasattr(model, "reset"):
            model.reset()

    # ------------------------------------------------------------
    # Experimental-control verification.
    # ------------------------------------------------------------
    aligned_state = aligned_model.state_dict()
    control_state = control_model.state_dict()

    if aligned_state.keys() != control_state.keys():
        raise RuntimeError("Aligned/control state_dict keys do not match.")

    for key in aligned_state:
        if not torch.equal(aligned_state[key], control_state[key]):
            raise RuntimeError(
                f"Aligned/control initialization differs at key: {key}"
            )

    # Also verify the models do not share tensor storage.
    aligned_param = next(aligned_model.parameters())
    control_param = next(control_model.parameters())

    if aligned_param.data_ptr() == control_param.data_ptr():
        raise RuntimeError(
            "Aligned/control models unexpectedly share parameter storage."
        )

    return aligned_model, control_model
