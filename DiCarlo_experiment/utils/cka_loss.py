"""
Differentiable linear CKA utilities for DynVision neural-alignment training.

This version follows the neural-alignment loss used in the DiCarlo et al.
(ICLR 2023) paper:

    L_neural = log(1 - CKA(X, Y))

The implementation uses B x B Gram matrices rather than P x P feature
covariance matrices, which is much more memory-friendly when convolutional
activations have very large feature dimensions.

Typical use:

    from cka_loss import linear_cka, cka_loss

    similarity = linear_cka(student_repr, target_repr)
    loss = cka_loss(student_repr, target_repr)

The returned loss is differentiable with respect to student_repr as long as
student_repr remains connected to autograd. The target representation may be
detached/frozen.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def _autocast_disabled(device: torch.device):
    """
    Context that turns autocast OFF for this device type.

    Why this exists
    ---------------
    Inside a `torch.autocast(...)` region, PyTorch re-casts the inputs of every
    matmul to the low-precision dtype (bf16/fp16), no matter what dtype the
    tensors had. So calling `.float()` before `x @ x.T` is NOT enough: the Gram
    matrix would still be computed and stored in bf16. CKA near 1 is exactly
    where log(1 - CKA) is most sensitive, so we compute it in float32.
    """
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except (RuntimeError, ValueError):
        # Device types without autocast support (e.g. MPS on older PyTorch)
        # are never autocast in the first place.
        return nullcontext()


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a representation tensor to [batch, features].

    Examples
    --------
    [B, P]          -> [B, P]
    [B, C, H, W]    -> [B, C*H*W]
    """
    if x.ndim < 2:
        raise ValueError(
            f"Expected representation with at least 2 dimensions, got {tuple(x.shape)}"
        )

    if x.ndim == 2:
        return x

    return x.flatten(start_dim=1)


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    """
    Center a square Gram matrix across samples.

    Equivalent to H K H, where
        H = I - (1/n) 11^T
    but implemented without explicitly constructing H.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(
            f"Expected a square Gram matrix, got shape {tuple(gram.shape)}"
        )

    row_mean = gram.mean(dim=1, keepdim=True)
    col_mean = gram.mean(dim=0, keepdim=True)
    grand_mean = gram.mean()

    return gram - row_mean - col_mean + grand_mean


def linear_cka(x: torch.Tensor, y: torch.Tensor, *,eps: float = 1e-8) -> torch.Tensor:
    """
    Compute differentiable linear CKA similarity between two representations.

    Parameters
    ----------
    x:
        First representation. Shape [B, P_x] or any [B, ...] tensor.
    y:
        Second representation. Shape [B, P_y] or any [B, ...] tensor.
        x and y may have different feature dimensions, but must contain the
        same samples in the same order.
    eps:
        Small numerical-stability constant used in the denominator.

    Returns
    -------
    torch.Tensor
        Scalar CKA similarity.

    Notes
    -----
    CKA is computed from centered linear-kernel Gram matrices:

        K = X X^T
        L = Y Y^T

        CKA = <K_c, L_c>_F
              -----------------------------
              ||K_c||_F ||L_c||_F

    where K_c and L_c are centered across samples.

    This B x B formulation avoids constructing enormous P x P matrices.

    Always computed in float32 with autocast disabled, even when called from
    inside a bf16/fp16 autocast region (see `_autocast_disabled`).
    """
    with _autocast_disabled(x.device):
        return _linear_cka_fp32(x, y, eps=eps)


def _linear_cka_fp32(x: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Body of `linear_cka`; must run with autocast disabled."""
    x = _flatten_features(x)
    y = _flatten_features(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must contain the same number of samples. "
            f"Got {x.shape[0]} and {y.shape[0]}."
        )

    if x.shape[0] < 2:
        raise ValueError(
            "CKA requires at least 2 samples in the batch. "
            f"Got batch size {x.shape[0]}."
        )

    # Upcast low-precision activations. Because autocast is disabled around
    # this function, the matmuls below now genuinely run in float32.
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if y.dtype in (torch.float16, torch.bfloat16):
        y = y.float()

    # Linear-kernel Gram matrices over samples: [B, B].
    gram_x = x @ x.T
    gram_y = y @ y.T

    gram_x = _center_gram(gram_x)
    gram_y = _center_gram(gram_y)

    # Frobenius inner product in numerator.
    numerator = (gram_x * gram_y).sum()

    # Product of Frobenius norms in denominator.
    norm_x = torch.linalg.vector_norm(gram_x)
    norm_y = torch.linalg.vector_norm(gram_y)
    denominator = norm_x * norm_y

    cka = numerator / denominator.clamp_min(eps)

    # Floating-point roundoff may place CKA slightly outside [0, 1].
    return cka.clamp(min=0.0, max=1.0)


def cka_loss(student_repr: torch.Tensor, target_repr: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the DiCarlo-style neural-alignment loss:

        L_neural = log(1 - CKA(student_repr, target_repr))

    Lower is better.

    Examples
    --------
    CKA = 0.50  -> loss = log(0.50)  ~= -0.693
    CKA = 0.90  -> loss = log(0.10)  ~= -2.303
    CKA = 0.99  -> loss = log(0.01)  ~= -4.605

    Parameters
    ----------
    student_repr:
        Student representation, typically [B, P] or [B, C, H, W].
        If connected to autograd, gradients flow back into the student model.
    target_repr:
        Target representation for the same samples, in the same order.
        This may be detached/frozen.
    eps:
        Lower bound on (1 - CKA) to avoid log(0) -> -inf.

    Returns
    -------
    torch.Tensor
        Scalar neural-alignment loss.
    """
    similarity = linear_cka(
        student_repr,
        target_repr,
        eps=eps,
    )

    one_minus_cka = (1.0 - similarity).clamp_min(eps)

    return torch.log(one_minus_cka)