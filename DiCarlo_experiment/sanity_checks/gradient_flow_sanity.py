
"""
Very lightweight gradient-flow sanity check for activation_utils.py.

Purpose:
Verify that activations captured by `capture_layer_responses(...)`
still belong to PyTorch's autograd graph, so a loss computed from them
can backpropagate into model parameters.

This is NOT a training run.
It uses:
- random initialization
- batch size = 1
- timesteps = 1
- image size = 32 x 32
- only one captured layer (V4)
- no checkpoint loading
- no optimizer step

Expected location:
DynVision/
└── DiCarlo_experiment/
    ├── activation_utils.py
    └── gradient_flow_sanity.py

Run from DynVision repo root:

    python DiCarlo_experiment/gradient_flow_sanity.py
"""

import torch
from dynvision.models import DyRCNNx4

from activation_utils import capture_layer_responses


print("=" * 72)
print("LIGHTWEIGHT GRADIENT-FLOW SANITY CHECK")
print("=" * 72)

# ------------------------------------------------------------
# 1. Build a SMALL randomly initialized trainable model
# ------------------------------------------------------------
#
# We intentionally use only 1 timestep and 32x32 input to keep
# memory usage very small. This test is only checking whether the
# captured activation remains connected to model parameters.

model = DyRCNNx4(
    n_classes=10,
    input_dims=(1, 3, 32, 32),
    n_timesteps=1,
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

model.train()

print("\n[OK] Small trainable DyRCNNx4 created.")

# ------------------------------------------------------------
# 2. Tiny dummy input
# ------------------------------------------------------------

x = torch.randn(
    (1, 1, 3, 32, 32),
    dtype=torch.float32,
)

print("Input shape:", tuple(x.shape))

# ------------------------------------------------------------
# 3. Capture ONE layer only
# ------------------------------------------------------------
#
# Important:
# DO NOT use torch.no_grad() here.
# We want autograd to track the computation.

logits, activations = capture_layer_responses(
    model,
    x,
    layers=("V4",),
    reset_state=True,
)

v4 = activations["V4"]

print("\nCaptured V4:")
print("  shape:", tuple(v4.shape))
print("  requires_grad:", v4.requires_grad)
print("  grad_fn:", type(v4.grad_fn).__name__ if v4.grad_fn is not None else None)

assert v4.requires_grad, (
    "FAIL: Captured V4 activation does not require gradients."
)

# ------------------------------------------------------------
# 4. Use a trivial scalar loss
# ------------------------------------------------------------
#
# This is NOT our future CKA loss.
# It is just the cheapest possible scalar function of the activation.

loss = v4.mean()

print("\nTemporary test loss:")
print("  value:", float(loss.detach()))
print("  requires_grad:", loss.requires_grad)

# ------------------------------------------------------------
# 5. Backpropagate once
# ------------------------------------------------------------

model.zero_grad(set_to_none=True)
loss.backward()

print("\n[OK] loss.backward() completed.")

# ------------------------------------------------------------
# 6. Check whether gradients reached real model parameters
# ------------------------------------------------------------

parameters_to_check = {
    "V1.conv.weight": model.V1.conv.weight,
    "V4.conv.weight": model.V4.conv.weight,
}

all_good = True

print("\nGradient checks:")

for name, parameter in parameters_to_check.items():
    grad = parameter.grad

    if grad is None:
        print(f"  {name:<20} grad=None   <-- FAIL")
        all_good = False
    else:
        finite = bool(torch.isfinite(grad).all())
        mean_abs = float(grad.abs().mean())
        max_abs = float(grad.abs().max())

        print(
            f"  {name:<20} "
            f"grad exists | "
            f"finite={finite} | "
            f"mean|grad|={mean_abs:.6e} | "
            f"max|grad|={max_abs:.6e}"
        )

        if not finite:
            all_good = False

# ------------------------------------------------------------
# 7. Final result
# ------------------------------------------------------------

print("\n" + "=" * 72)

if all_good:
    print("FULL SUCCESS")
    print(
        "The captured V4 activation stayed connected to autograd, "
        "and gradients propagated back into model weights."
    )
else:
    print("FAILED / NEEDS REVIEW")
    print(
        "At least one checked model parameter did not receive a valid gradient."
    )

print("=" * 72)

# Optional cleanup for low-memory machines.
del logits, activations, v4, loss, x, model
