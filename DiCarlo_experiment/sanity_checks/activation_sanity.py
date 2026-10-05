
"""
Sanity check for differentiable DynVision activation extraction.
Using the activation_utils.py helper
Expected structure:

Run from the DynVision repo root:

    python DiCarlo_experiment/activation_sanity.py
"""

from pathlib import Path

import torch
from dynvision.models import DyRCNNx4

from activation_utils import capture_layer_responses


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    SCRIPT_DIR
    / "x4_generators_tfb12"
    / "x4_pointdepthwise_tfb12"
    / "trained.pt"
)

LAYERS = ("V1", "V2", "V4", "IT")


def build_target_model():
    """Reconstruct the target architecture exactly as in our verified sanity check."""
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


def load_target_model():
    """
    Fully initialize lazy skip/feedback adapters, then strictly load trained.pt.
    """
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found:\n{CHECKPOINT_PATH}"

    model = build_target_model()
    model.eval()

    # We already verified that a full 20-timestep probe is required to create
    # both skip and delayed-feedback adapter modules for this target.
    probe = torch.zeros((1, 20, 3, 224, 224), dtype=torch.float32)

    with torch.no_grad():
        _ = model(probe, store_responses=False)

    try:
        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    assert model_keys == checkpoint_keys, (
        "Model/checkpoint keys do not match.\n"
        f"Checkpoint-only: {sorted(checkpoint_keys - model_keys)}\n"
        f"Model-only: {sorted(model_keys - checkpoint_keys)}"
    )

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


print("=" * 82)
print("DYNVISION DIFFERENTIABLE ACTIVATION HELPER SANITY CHECK")
print("=" * 82)

print("\n1. Loading verified trained target model...")
target_model = load_target_model()
print("[OK] Target model loaded and frozen.")

print("\n2. Creating a 20-timestep dummy input...")
x = torch.zeros((1, 20, 3, 224, 224), dtype=torch.float32)
print("Input shape:", tuple(x.shape))

print("\n3. Capturing V1 / V2 / V4 / IT responses in ONE forward pass...")

with torch.no_grad():
    logits, activations = capture_layer_responses(
        target_model,
        x,
        layers=LAYERS,
        reset_state=True,
    )

print("[OK] Forward + activation capture succeeded.")

print("\nClassifier output:")
print("  logits shape:", tuple(logits.shape))

print("\nCaptured visual-layer activations:")
for layer in LAYERS:
    a = activations[layer]
    print(
        f"  {layer:<3} shape={tuple(a.shape)} "
        f"dtype={a.dtype} "
        f"device={a.device} "
        f"requires_grad={a.requires_grad}"
    )

print("\n4. Checking temporal dimension...")
assert logits.shape[1] == 20, (
    f"Expected 20 classifier timesteps, got {logits.shape[1]}"
)
for layer in LAYERS:
    assert activations[layer].shape[1] == 20, (
        f"{layer} did not contain 20 captured timesteps."
    )
print("[OK] All outputs contain 20 real timesteps.")

print("\n5. Example: take IT at the LAST timestep...")
it_last = activations["IT"][:, -1, ...]
print("  IT last-timestep shape:", tuple(it_last.shape))

print("\n6. Example: flatten IT into a CKA representation matrix...")
it_last_flat = it_last.flatten(start_dim=1)
print("  Flattened IT shape:", tuple(it_last_flat.shape))
print("  Interpretation: [batch/images, features]")

print("\n7. Checking target-side gradient behavior...")
print("  IT requires_grad:", activations["IT"].requires_grad)
assert activations["IT"].requires_grad is False
print("[OK] Target activations have no gradient, as expected under torch.no_grad().")

print("\n" + "=" * 82)
print("FULL SUCCESS")
print("=" * 82)
print(
    "\nThis verifies that the helper can extract V1/V2/V4/IT activations "
    "across all 20 timesteps from the fully loaded target model."
)
print(
    "\nNext, the same helper can be used WITHOUT torch.no_grad() on a trainable "
    "student model so its captured activations retain autograd history for CKA."
)
