"""
Utilities for extracting differentiable intermediate responses from DynVision models.

The key idea:
- DynVision's public temporal forward loop detaches recorded responses before
  storing/offloading them.
- The per-timestep `_forward(...)` method returns `responses_t` before that
  detach happens.
- This helper temporarily wraps `_forward`, captures selected layer responses,
  then restores the original method.

This keeps the activation computation graph intact for trainable/student models.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch


def _resolve_timesteps(
    timesteps: Optional[Sequence[int]],
    n_timesteps: int,
) -> Optional[Tuple[int, ...]]:
    """
    Turn requested timesteps (negative indices allowed, e.g. -1 = last) into
    non-negative indices over the REAL (non-idle) timesteps.
    """
    if timesteps is None:
        return None

    resolved = []
    for t in timesteps:
        t_pos = t + n_timesteps if t < 0 else t
        if not 0 <= t_pos < n_timesteps:
            raise IndexError(
                f"Requested timestep {t} is out of range for "
                f"{n_timesteps} input timesteps."
            )
        resolved.append(t_pos)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate timesteps requested: {list(timesteps)}")

    return tuple(resolved)


def capture_layer_responses(model, x: torch.Tensor,
                            layers: Iterable[str] = ("V4",), *,reset_state: bool = True,
                            timesteps: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run a DynVision model and capture selected layer activations across time.

    Parameters
    ----------
    model:
        A DynVision temporal model exposing `_forward(...)`.
    x:
        Input tensor shaped [B, T, C, H, W].
    layers:
        Layer names to capture, e.g. ("V4",) or ("V1", "V2", "V4", "IT").
    reset_state:
        If True and the model exposes `reset()`, clear temporal state before
        the forward pass.
    timesteps:
        Optional real-timestep indices to keep, e.g. (-1,) for the last one.
        Negative indices count from the end. Idle timesteps are never counted.
        None (default) keeps every real timestep, as before.

        Keeping only what the loss needs avoids building a [B, T, ...] stacked
        copy of the layer, and for no-grad models (the target) lets the
        unused timesteps' activations be freed immediately.

    Returns
    -------
    logits:
        Model output, typically [B, T, n_classes]. Always ALL timesteps.
    activations:
        Dictionary mapping each requested layer to a tensor stacked over time.
        For convolutional visual layers this is typically [B, T, C, H, W].
        If `timesteps` is given, dim 1 has len(timesteps) entries, in the
        order requested (so `timesteps=(-1,)` -> select index 0).

    Notes
    -----
    This function intentionally calls the *outer* model forward with
    `store_responses=False` so DynVision does not CPU-offload/detach the
    activations. Inside the wrapped per-timestep `_forward`, response recording
    is forced on so we can capture the raw `responses_t` tensors.

    If the model is trainable and the caller is NOT inside `torch.no_grad()`,
    the returned activations retain autograd history and can be used in a loss.
    """
    layers = tuple(layers)

    if x.ndim != 5:
        raise ValueError(
            f"Expected x with shape [B, T, C, H, W], but got {tuple(x.shape)}"
        )

    if not hasattr(model, "_forward"):
        raise AttributeError(
            "Model does not expose a `_forward` method; this helper is intended "
            "for DynVision temporal models."
        )

    if reset_state and hasattr(model, "reset"):
        model.reset()

    keep = _resolve_timesteps(timesteps, n_timesteps=x.shape[1])

    # Real timestep index -> {layer: tensor}. Filled by the wrapper below.
    captured_by_t: Dict[int, Dict[str, torch.Tensor]] = {}
    real_t = 0  # counts only non-idle timesteps
    original_forward = model._forward

    def capturing_forward(x_t=None, *args, **kwargs):
        nonlocal real_t

        # Force the single-timestep forward to populate responses_t.
        kwargs["store_responses"] = True

        out, responses_t = original_forward(x_t, *args, **kwargs)

        # Idle timesteps (if configured) should not be mixed into the requested
        # real stimulus timesteps. DynVision sets this flag during its idle loop.
        in_idle_period = getattr(model, "_in_idle_period", False)

        if not in_idle_period:
            t_now = real_t
            real_t += 1

            if keep is not None and t_now not in keep:
                return out, responses_t

            captured_by_t[t_now] = {}
            for layer in layers:
                if layer not in responses_t:
                    available = sorted(responses_t.keys())
                    raise KeyError(
                        f"Requested layer '{layer}' was not present in responses_t. "
                        f"Available keys: {available}"
                    )
                captured_by_t[t_now][layer] = responses_t[layer]

        return out, responses_t

    model._forward = capturing_forward

    try:
        # NOTE: DynVision's outer forward() ignores this kwarg; whether it also
        # CPU-offloads a detached copy of responses is decided by
        # model.storage. Either way, the tensors captured above are the
        # original (non-detached) ones, so gradients are unaffected.
        logits = model(x, store_responses=False)
    finally:
        # Always restore the original model method, even if forward raises.
        model._forward = original_forward

    order = keep if keep is not None else tuple(sorted(captured_by_t))
    missing = [t for t in order if t not in captured_by_t]
    if missing or not order:
        raise RuntimeError(
            f"No activations were captured for timesteps {missing or 'any'} "
            f"(captured: {sorted(captured_by_t)})."
        )

    activations: Dict[str, torch.Tensor] = {}

    for layer in layers:
        # Each per-timestep tensor is typically [B, C, H, W].
        # Stack at dim=1 -> [B, T_kept, C, H, W], in the requested order.
        activations[layer] = torch.stack(
            [captured_by_t[t][layer] for t in order],
            dim=1,
        )

    return logits, activations


def disable_response_storage(model) -> None:
    """
    Turn off DynVision's built-in response recording for this model.

    By default every forward pass copies EVERY layer's activity at EVERY
    timestep to the CPU (DynVision's StorageBuffer: a "cyclic" buffer reports
    should_store() == True even with capacity 0). DynVision's own Lightning
    training disables this during training; called directly, the model keeps
    it on, which makes each batch many times slower (tens of GB copied
    GPU -> CPU per batch for our three models).

    capture_layer_responses() does not depend on this buffer, so model
    outputs and captured activations are unchanged.
    """
    if not hasattr(model, "storage"):
        return
    from dynvision.base.storage import StorageBuffer

    model.storage = StorageBuffer(
        max_responses=0,
        max_records=0,
        response_strategy="fixed",
        record_strategy="fixed",
    )
    if model.storage.responses.should_store():
        raise RuntimeError("Could not disable DynVision response storage.")