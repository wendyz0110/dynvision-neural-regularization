"""
Sanity check for fully reconstructing and loading the trained DynVision target model.

Expected structure:

DynVision/
├── DiCarlo_experiment/
│   ├── target_model_sanity.py
│   └── x4_generators_tfb12/
│       └── x4_pointdepthwise_tfb12/
│           ├── init.pt
│           ├── init.pt.config.yaml
│           ├── trained.config.yaml
│           ├── trained.pt
│           └── trained.pt.config.yaml
├── dynvision/
└── ...

Run from the DynVision repo root:

    python DiCarlo_experiment/target_model_sanity.py

This version forces a 20-timestep dummy forward BEFORE loading the checkpoint,
so that delayed feedback adapter modules have a chance to be lazily created.
"""

from pathlib import Path

import torch
from dynvision.models import DyRCNNx4


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TARGET_DIR = (
    SCRIPT_DIR
    / "x4_generators_tfb12"
    / "x4_pointdepthwise_tfb12"
)

CHECKPOINT_PATH = TARGET_DIR / "trained.pt"
CONFIG_PATH = TARGET_DIR / "trained.pt.config.yaml"

print("=" * 78)
print("TARGET MODEL SANITY CHECK")
print("=" * 78)

print("\nScript directory:")
print(SCRIPT_DIR)

print("\nTarget directory:")
print(TARGET_DIR)

print("\nCheckpoint path:")
print(CHECKPOINT_PATH)

assert TARGET_DIR.exists(), f"Target directory not found:\n{TARGET_DIR}"
assert CHECKPOINT_PATH.exists(), f"Checkpoint not found:\n{CHECKPOINT_PATH}"
assert CONFIG_PATH.exists(), f"Config not found:\n{CONFIG_PATH}"

print("\n[OK] Target files found.")


# ============================================================
# 2. Reconstruct target architecture
# ============================================================

print("\nConstructing DyRCNNx4 target model...")

target_model = DyRCNNx4(
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

print("[OK] Model instantiated.")
print("Model class:", target_model.__class__.__name__)


# ============================================================
# 3. Read checkpoint
# ============================================================

print("\nReading trained.pt...")

try:
    checkpoint_state = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    checkpoint_state = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

print("[OK] Checkpoint read.")
print("Checkpoint object type:", type(checkpoint_state))
print("Number of checkpoint entries:", len(checkpoint_state))


# ============================================================
# 4. Adapter parameters of interest
# ============================================================

ADAPTER_KEYS = [
    "addskip_V4.conv.weight",
    "addskip_V4.conv.bias",
    "addfeedback_V1.conv.weight",
    "addfeedback_V1.conv.bias",
    "addskip_IT.conv.weight",
    "addskip_IT.conv.bias",
    "addfeedback_V2.conv.weight",
    "addfeedback_V2.conv.bias",
]

print("\nChecking adapter parameters in checkpoint:")

for key in ADAPTER_KEYS:
    print(f"  {key:<40} {'FOUND' if key in checkpoint_state else 'MISSING'}")


# ============================================================
# 5. Before dummy forward
# ============================================================

fresh_keys = set(target_model.state_dict().keys())

print("\nBefore dummy forward, adapter parameters present in model:")

for key in ADAPTER_KEYS:
    print(f"  {key:<40} {'YES' if key in fresh_keys else 'NO'}")


# ============================================================
# 6. Force a full 20-timestep dummy forward
# ============================================================
#
# This model was trained with n_timesteps=20.
# A 1-timestep forward is not enough to trigger feedback modules
# with t_feedback = 12 ms and dt = 2 ms.
#
# So we explicitly provide 20 temporal input steps:
#   [batch, time, channels, height, width]
# = [1, 20, 3, 224, 224]

print("\nRunning 20-timestep dummy forward BEFORE loading trained weights...")

target_model.eval()

dummy_input = torch.zeros(
    (1, 20, 3, 224, 224),
    dtype=torch.float32,
)

with torch.no_grad():
    dummy_output = target_model(dummy_input)

print("[OK] Dummy forward succeeded.")

if hasattr(dummy_output, "shape"):
    print("Dummy output shape:", tuple(dummy_output.shape))
else:
    print("Dummy output type:", type(dummy_output))


# ============================================================
# 7. After dummy forward
# ============================================================

post_forward_keys = set(target_model.state_dict().keys())

print("\nAfter 20-timestep dummy forward, adapter parameters present in model:")

all_adapters_created = True

for key in ADAPTER_KEYS:
    exists = key in post_forward_keys
    all_adapters_created &= exists
    print(f"  {key:<40} {'YES' if exists else 'NO'}")

if all_adapters_created:
    print("\n[OK] All 8 skip/feedback adapter parameters now exist.")
else:
    print("\n[WARNING] Some adapter parameters are still missing.")


# ============================================================
# 8. Compare model and checkpoint keys
# ============================================================

checkpoint_keys = set(checkpoint_state.keys())
model_keys = set(target_model.state_dict().keys())

checkpoint_only = sorted(checkpoint_keys - model_keys)
model_only = sorted(model_keys - checkpoint_keys)

print("\nKey comparison immediately before load_state_dict:")

print(f"  Checkpoint entries: {len(checkpoint_keys)}")
print(f"  Model entries:      {len(model_keys)}")
print(f"  Checkpoint-only:     {len(checkpoint_only)}")
print(f"  Model-only:          {len(model_only)}")

if checkpoint_only:
    print("\nKeys in checkpoint but NOT in model:")
    for key in checkpoint_only:
        print(" ", key)

if model_only:
    print("\nKeys in model but NOT in checkpoint:")
    for key in model_only:
        print(" ", key)


# ============================================================
# 9. Strict load
# ============================================================

print("\nStrictly loading trained.pt AFTER 20-timestep dummy forward...")

load_result = target_model.load_state_dict(
    checkpoint_state,
    strict=True,
)

print("\nload_state_dict returned:")
print(load_result)


# ============================================================
# 10. Freeze target model
# ============================================================

target_model.eval()

for parameter in target_model.parameters():
    parameter.requires_grad_(False)

assert not target_model.training
assert all(
    not parameter.requires_grad
    for parameter in target_model.parameters()
)


# ============================================================
# 11. Final report
# ============================================================

print("\n" + "=" * 78)

if all_adapters_created and not checkpoint_only and not model_only:
    print("FULL SUCCESS")
    print(
        "All skip/feedback adapter modules were created, "
        "model and checkpoint key sets match exactly, "
        "and strict loading completed."
    )
else:
    print("PARTIAL SUCCESS / NEEDS REVIEW")
    print(
        "The checkpoint loaded, but inspect the adapter/module warnings "
        "and key differences above before treating this as an exact reconstruction."
    )

print("=" * 78)

print("\n[OK] Target model is in eval mode.")
print("[OK] All target parameters are frozen.")
print("\nNo training and no GPU are required for this test.")
