"""
Utilities for reconstructing and loading the trained DynVision target model.

This module centralizes the target-model loading logic that was previously
repeated across sanity-check scripts.

Typical use:

    from target_model_utils import load_target_model

    target_model = load_target_model()

The returned model is:
- reconstructed with the verified DyRCNNx4 configuration
- initialized with a 20-timestep probe so lazy skip/feedback adapters exist
- strictly loaded from trained.pt
- moved to the requested device
- placed in eval mode
- frozen by default
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from dynvision.models import DyRCNNx4


# ---------------------------------------------------------------------
# Default checkpoint location, resolved relative to this file.
# ---------------------------------------------------------------------

UTILS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = UTILS_DIR.parent

DEFAULT_CHECKPOINT_PATH = (
    EXPERIMENT_DIR
    / "x4_generators_tfb12"
    / "x4_pointdepthwise_tfb12"
    / "trained.pt"
)


def build_target_model() -> DyRCNNx4:
    """
    Reconstruct the verified DyRCNNx4 target architecture.

    Returns
    -------
    DyRCNNx4
        Randomly initialized model with the same architecture/configuration
        as the trained target checkpoint.
    """
    model = DyRCNNx4(
        n_classes=10,
        input_dims=(1, 3, 224, 224),
        n_timesteps=20,
        dt=2,
        tau=9,
        t_feedforward=0,
        t_recurrence=6,
        t_feedback=12,
        t_skip=0,
        recurrence_type="pointdepthwise",
        recurrence_target="output",
        skip=True,
        feedback="add",
    )

    return model


def initialize_lazy_connections(
    model: DyRCNNx4,
    *,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Run a 20-timestep dummy forward to instantiate lazy skip/feedback adapters.

    For this specific target configuration, a 1-timestep probe is insufficient
    because delayed feedback modules are not created until later timesteps.

    Parameters
    ----------
    model:
        Reconstructed DyRCNNx4 model.
    device:
        Device on which to run the initialization probe.
    """
    device = torch.device(device)

    model.to(device)
    model.eval()

    probe = torch.zeros(
        (1, 20, 3, 224, 224),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        _ = model(probe, store_responses=False)

    # Clear temporal state left over from the probe.
    if hasattr(model, "reset"):
        model.reset()


def load_target_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    verbose: bool = False,
) -> DyRCNNx4:
    """
    Reconstruct and strictly load the trained DynVision target model.

    Parameters
    ----------
    checkpoint_path:
        Path to trained.pt. If omitted, uses the verified project-relative path:
        DiCarlo_experiment/x4_generators_tfb12/x4_pointdepthwise_tfb12/trained.pt
    device:
        Device for the returned model, e.g. "cpu", "cuda", or "mps".
    freeze:
        If True, set requires_grad=False for all target parameters.
    verbose:
        If True, print loading diagnostics.

    Returns
    -------
    DyRCNNx4
        Fully reconstructed and loaded target model.
    """
    device = torch.device(device)

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Target checkpoint not found:\n{checkpoint_path}"
        )

    if verbose:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device:     {device}")
        print("Constructing DyRCNNx4...")

    model = build_target_model()

    # Important: create all lazy skip/feedback adapter modules before loading.
    if verbose:
        print("Initializing lazy skip/feedback adapters...")

    initialize_lazy_connections(model, device=device)

    if verbose:
        print("Reading checkpoint...")

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )

    # Explicitly verify exact key agreement before load_state_dict().
    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())

    if checkpoint_keys != model_keys:
        checkpoint_only = sorted(checkpoint_keys - model_keys)
        model_only = sorted(model_keys - checkpoint_keys)

        raise RuntimeError(
            "Target checkpoint and reconstructed model do not match exactly.\n"
            f"Checkpoint-only keys: {checkpoint_only}\n"
            f"Model-only keys: {model_only}"
        )

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    # Ensure no temporal state from initialization survives into real use.
    if hasattr(model, "reset"):
        model.reset()

    if verbose:
        print(f"[OK] Loaded {len(state_dict)} state_dict entries.")
        print("[OK] Model is in eval mode.")
        print(f"[OK] Frozen: {freeze}")

    return model
