
"""
Pseudo-label helpers for target-model supervision.

The target model produces logits across time:
    [B, T, n_classes]

For the current MVP, the classification target is the target model's
prediction at one chosen timestep (default: final timestep), NOT the
Imagenette ground-truth label.
"""

from __future__ import annotations

from typing import Dict, Union

import torch


def extract_pseudo_labels(
    logits: torch.Tensor,
    *,
    timestep: int = -1,
) -> torch.Tensor:
    """
    Convert target-model logits into hard pseudo-labels.

    Parameters
    ----------
    logits:
        Either:
        - [B, T, C] temporal classifier logits, or
        - [B, C] already-selected logits.
    timestep:
        Timestep to use when logits are [B, T, C].
        Default -1 means final timestep.

    Returns
    -------
    torch.Tensor
        Integer class indices with shape [B].

    Notes
    -----
    No softmax is needed because argmax(logits) == argmax(softmax(logits)).
    """
    if logits.ndim == 3:
        selected_logits = logits[:, timestep, :]
    elif logits.ndim == 2:
        selected_logits = logits
    else:
        raise ValueError(
            "Expected logits shaped [B, T, C] or [B, C], "
            f"got {tuple(logits.shape)}."
        )

    return selected_logits.argmax(dim=-1)


def get_pseudo_label_outputs(
    logits: torch.Tensor,
    *,
    timestep: int = -1,
    include_probabilities: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper returning selected logits + pseudo-labels.

    Useful for debugging/evaluation when you want more than just labels.
    """
    if logits.ndim == 3:
        selected_logits = logits[:, timestep, :]
    elif logits.ndim == 2:
        selected_logits = logits
    else:
        raise ValueError(
            "Expected logits shaped [B, T, C] or [B, C], "
            f"got {tuple(logits.shape)}."
        )

    result = {
        "logits": selected_logits,
        "labels": selected_logits.argmax(dim=-1),
    }

    if include_probabilities:
        result["probabilities"] = torch.softmax(selected_logits, dim=-1)

    return result
