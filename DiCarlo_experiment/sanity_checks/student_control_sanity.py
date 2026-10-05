
"""
Lightweight sanity check for:
1. flexible student construction,
2. identical aligned/control initialization,
3. target pseudo-label extraction.

Run:
    python DiCarlo_experiment/student_control_sanity.py
"""

import torch
from dynvision.models import DyRCNNx4

from student_model_utils import (
    build_student_model,
    create_aligned_control_pair,
)
from pseudo_label_utils import extract_pseudo_labels


print("=" * 76)
print("STUDENT / CONTROL / PSEUDO-LABEL SANITY CHECK")
print("=" * 76)

tiny_overrides = {
    "input_dims": (1, 3, 32, 32),
    "n_timesteps": 1,
}

student = build_student_model(
    model_cls=DyRCNNx4,
    overrides=tiny_overrides,
    seed=123,
)

print("\n[OK] Flexible student model built.")
print("Temporary sanity-test overrides:")
print("  input_dims =", tiny_overrides["input_dims"])
print("  n_timesteps =", tiny_overrides["n_timesteps"])


aligned_model, control_model = create_aligned_control_pair(
    model_cls=DyRCNNx4,
    overrides=tiny_overrides,
    seed=123,
    initialize_lazy=False,
)

all_equal = all(
    torch.equal(aligned_model.state_dict()[k], control_model.state_dict()[k])
    for k in aligned_model.state_dict()
)

assert all_equal

print("\n[OK] Aligned and control models have identical initial state_dicts.")

first_aligned_param = next(aligned_model.parameters())
first_control_param = next(control_model.parameters())

assert first_aligned_param.data_ptr() != first_control_param.data_ptr()

print("[OK] They are independent model objects with separate parameter storage.")


# Fake temporal target logits:
# B=3, T=2, C=4
target_logits = torch.tensor(
    [
        [[0.1, 2.0, 0.3, 0.4], [0.2, 0.1, 4.0, 0.3]],
        [[1.5, 0.1, 0.2, 0.3], [0.1, 3.0, 0.2, 0.4]],
        [[0.1, 0.2, 0.3, 2.0], [5.0, 0.2, 0.1, 0.0]],
    ],
    dtype=torch.float32,
)

labels = extract_pseudo_labels(target_logits, timestep=-1)

print("\nFinal-timestep target pseudo-labels:")
print(labels)

expected = torch.tensor([2, 1, 0])
assert torch.equal(labels, expected)

print("[OK] Pseudo-label extraction uses final-timestep argmax correctly.")

print("\n" + "=" * 76)
print("FULL SUCCESS")
print("=" * 76)
