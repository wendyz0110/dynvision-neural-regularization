
"""
Lightweight sanity checks for cka_loss.py.

This script tests three things:

1. Identical representations should have CKA ~= 1.
2. Unrelated random representations should have lower CKA.
3. Dapello-style log(1 - CKA) loss should backpropagate a finite,
   non-zero gradient into the student representation.

This does NOT load DynVision or any checkpoint, so it is extremely lightweight.

Run from the DynVision repo root:

    python DiCarlo_experiment/cka_sanity.py
"""

import math
import torch

from cka_loss_utils import linear_cka, cka_loss


torch.manual_seed(0)

BATCH = 32
FEATURES = 64

print("=" * 72)
print("CKA LOSS SANITY CHECK")
print("=" * 72)


# ---------------------------------------------------------------------
# Test 1: identical representations
# ---------------------------------------------------------------------

x = torch.randn(BATCH, FEATURES)

cka_same = linear_cka(x, x)
loss_same = cka_loss(x, x)

print("\n1. IDENTICAL REPRESENTATIONS")
print(f"CKA(x, x):          {cka_same.item():.6f}")
print(f"log(1 - CKA):       {loss_same.item():.6f}")

assert torch.isfinite(cka_same), "CKA(x, x) is not finite."
assert abs(cka_same.item() - 1.0) < 1e-5, (
    f"Expected CKA(x, x) ~= 1, got {cka_same.item()}"
)

print("[OK] Identical representations give CKA ~= 1.")


# ---------------------------------------------------------------------
# Test 2: unrelated random representations
# ---------------------------------------------------------------------

y = torch.randn(BATCH, FEATURES)

cka_random = linear_cka(x, y)
loss_random = cka_loss(x, y)

print("\n2. UNRELATED RANDOM REPRESENTATIONS")
print(f"CKA(x, y):          {cka_random.item():.6f}")
print(f"log(1 - CKA):       {loss_random.item():.6f}")

assert torch.isfinite(cka_random), "Random CKA is not finite."
assert cka_random < cka_same, (
    "Expected unrelated representations to have lower CKA "
    "than identical representations."
)

print("[OK] Random representations have lower CKA than identical ones.")


# ---------------------------------------------------------------------
# Test 3: gradient flow through log(1 - CKA)
# ---------------------------------------------------------------------
#
# Do NOT use identical representations here:
# if CKA is exactly 1, the epsilon clamp can intentionally flatten the loss.
#
# Instead, make target related-but-not-identical to student.

student = torch.randn(BATCH, FEATURES, requires_grad=True)

noise = torch.randn(BATCH, FEATURES)
target = 0.8 * student.detach() + 0.2 * noise

similarity = linear_cka(student, target)
loss = cka_loss(student, target)

print("\n3. GRADIENT FLOW")
print(f"CKA(student, target): {similarity.item():.6f}")
print(f"log(1 - CKA):         {loss.item():.6f}")
print(f"loss.requires_grad:    {loss.requires_grad}")

assert loss.requires_grad, "Loss is not connected to autograd."

loss.backward()

grad = student.grad

assert grad is not None, "No gradient reached student representation."
assert torch.isfinite(grad).all(), "Student gradient contains NaN/Inf."

mean_abs_grad = grad.abs().mean().item()
max_abs_grad = grad.abs().max().item()

print(f"mean|grad|:            {mean_abs_grad:.6e}")
print(f"max|grad|:             {max_abs_grad:.6e}")

assert max_abs_grad > 0.0, "Gradient exists but is entirely zero."

print("[OK] Neural-alignment loss produces a finite, non-zero gradient.")


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("\n" + "=" * 72)
print("FULL SUCCESS")
print("=" * 72)

print(
    "\nVerified:\n"
    "  - identical representations -> CKA ~= 1\n"
    "  - unrelated representations -> lower CKA\n"
    "  - log(1 - CKA) remains differentiable for student representations"
)
