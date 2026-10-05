"""
Minimal CPU smoke test for train_alignment.py.

Purpose
-------
Test the TRAINING ENGINE only, without:
- real image data
- real DynVision models
- real checkpoints
- GPU
- mixed precision
- real activation hooks / storage buffers

This checks that:
1. dummy data can enter run_epoch()
2. [B,C,H,W] images are expanded to [B,T,C,H,W]
3. target produces logits + representations
4. pseudo-labels are generated
5. aligned model computes CE + CKA loss
6. control model computes CE-only loss
7. backward + optimizer.step() work
8. validation works
9. metrics are finite
10. aligned/control start from identical weights
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train_alignment as ta

# ============================================================
# 1. Tiny mock temporal model
# ============================================================

class TinyTemporalModel(nn.Module):
    """
    Minimal stand-in for a temporal DynVision model.

    Input
    -----
    x: [B, T, C, H, W]

    Output
    ------
    logits: [B, T, n_classes]

    activations:
        {
            "V4": [B, T, hidden_dim]
        }
    """

    def __init__(
        self,
        image_shape=(3, 8, 8),
        hidden_dim=8,
        n_classes=5,
    ):
        super().__init__()

        c, h, w = image_shape
        input_dim = c * h * w

        self.feature = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            n_classes,
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(x.shape)}"
            )

        batch_size, n_timesteps, c, h, w = x.shape

        # Treat each timestep independently.
        x = x.reshape(
            batch_size * n_timesteps,
            c * h * w,
        )

        hidden = torch.relu(
            self.feature(x)
        )

        logits = self.classifier(
            hidden
        )

        # Restore temporal dimension.
        hidden = hidden.reshape(
            batch_size,
            n_timesteps,
            -1,
        )

        logits = logits.reshape(
            batch_size,
            n_timesteps,
            -1,
        )

        activations = {
            "V4": hidden
        }

        return logits, activations


# ============================================================
# 2. Mock DynVision activation-capture function
# ============================================================

def mock_capture_layer_responses(
    model,
    model_input,
    layers,
    reset_state=True,
):
    """
    Replacement for the real DynVision-specific
    capture_layer_responses().

    run_epoch() expects:

        logits, activations

    so our tiny model returns exactly that interface.
    """

    logits, activations = model(model_input)

    # Make sure requested layers exist.
    requested_activations = {}

    for layer in layers:
        if layer not in activations:
            raise KeyError(
                f"Mock model has no layer '{layer}'. "
                f"Available: {list(activations.keys())}"
            )

        requested_activations[layer] = activations[layer]

    return logits, requested_activations

# Patch only inside train_alignment module.
ta.capture_layer_responses = mock_capture_layer_responses

# ============================================================
# 3. Logger
# ============================================================

def build_logger():
    logger = logging.getLogger("alignment_sanity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ============================================================
# 4. Utility assertions
# ============================================================

def models_are_identical(model_a, model_b):
    """
    Check exact equality of all model parameters.
    """

    for param_a, param_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        if not torch.equal(param_a, param_b):
            return False

    return True


def model_changed(model, initial_state):
    """
    Return True if at least one parameter changed.
    """

    current_state = model.state_dict()

    for key in initial_state:
        if not torch.equal(
            current_state[key],
            initial_state[key],
        ):
            return True

    return False


def assert_finite_metrics(metrics):
    """
    Check all important scalar metrics are finite.
    """

    metric_names = (
        "total_loss",
        "ce_loss",
        "alignment_loss",
        "cka",
        "target_agreement",
        "gt_accuracy",
    )

    for model_name in ("aligned", "control"):

        assert model_name in metrics

        for metric_name in metric_names:

            assert metric_name in metrics[model_name]

            value = metrics[model_name][metric_name]

            tensor_value = torch.tensor(
                value,
                dtype=torch.float32,
            )

            assert torch.isfinite(tensor_value), (
                f"Non-finite metric: "
                f"{model_name}.{metric_name}={value}"
            )


# ============================================================
# 5. Main sanity check
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("Alignment pipeline CPU sanity check")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(123)

    device = torch.device("cpu")

    logger = build_logger()

    # --------------------------------------------------------
    # Minimal experiment config
    # --------------------------------------------------------

    cfg = ta.ExperimentConfig()

    cfg.runtime.device = "cpu"
    cfg.runtime.precision = "float32"

    # Keep temporal dimension tiny for CPU smoke test.
    cfg.data.input_timesteps = 3
    cfg.data.save_epoch_sample_order = False

    cfg.alignment.layer = "V4"
    cfg.alignment.timestep = -1

    cfg.alignment.ce_weight = 1.0
    cfg.alignment.cka_weight = 0.1

    cfg.alignment.cka_eps = 1e-6
    cfg.alignment.warmup_epochs = 0

    cfg.target.pseudo_label_timestep = -1

    cfg.optimization.accumulate_grad_batches = 1
    cfg.optimization.max_grad_norm = None

    cfg.runtime.log_every_n_steps = 1

    # --------------------------------------------------------
    # Dummy dataset
    # --------------------------------------------------------

    n_samples = 16
    n_classes = 5

    images = torch.randn(
        n_samples,
        3,
        8,
        8,
    )

    # These are only used for gt_accuracy monitoring.
    labels = torch.randint(
        low=0,
        high=n_classes,
        size=(n_samples,),
    )

    sample_ids = torch.arange(
        n_samples
    )

    dataset = TensorDataset(
        images,
        labels,
        sample_ids,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    print(
        f"\nDummy images shape: {tuple(images.shape)}"
    )

    print(
        "Expected temporal shape inside run_epoch: "
        f"({4}, {cfg.data.input_timesteps}, 3, 8, 8)"
    )

    # --------------------------------------------------------
    # Create target / aligned / control
    # --------------------------------------------------------

    target_model = TinyTemporalModel(
        hidden_dim=8,
        n_classes=n_classes,
    ).to(device)

    aligned_model = TinyTemporalModel(
        hidden_dim=8,
        n_classes=n_classes,
    ).to(device)

    # IMPORTANT:
    # control starts from EXACTLY the same weights.
    control_model = copy.deepcopy(
        aligned_model
    ).to(device)

    assert models_are_identical(
        aligned_model,
        control_model,
    )

    print(
        "\n[PASS] aligned/control initial weights are identical"
    )

    # --------------------------------------------------------
    # Freeze target
    # --------------------------------------------------------

    for parameter in target_model.parameters():
        parameter.requires_grad = False

    target_model.eval()

    assert all(
        not parameter.requires_grad
        for parameter in target_model.parameters()
    )

    print(
        "[PASS] target model is frozen"
    )

    # Save parameter snapshots for later checks.
    aligned_initial_state = copy.deepcopy(
        aligned_model.state_dict()
    )

    control_initial_state = copy.deepcopy(
        control_model.state_dict()
    )

    target_initial_state = copy.deepcopy(
        target_model.state_dict()
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    aligned_optimizer = torch.optim.Adam(
        aligned_model.parameters(),
        lr=1e-3,
    )

    control_optimizer = torch.optim.Adam(
        control_model.parameters(),
        lr=1e-3,
    )

    # ========================================================
    # TRAIN EPOCH
    # ========================================================

    print("\n" + "-" * 70)
    print("Running TRAIN epoch")
    print("-" * 70)

    train_metrics = ta.run_epoch(
        split="train",
        epoch=0,
        loader=train_loader,
        target_model=target_model,
        aligned_model=aligned_model,
        control_model=control_model,
        aligned_optimizer=aligned_optimizer,
        control_optimizer=control_optimizer,
        cfg=cfg,
        device=device,
        logger=logger,
        sample_order_path=None,
    )

    print("\nTrain metrics:")
    print(train_metrics)

    assert_finite_metrics(
        train_metrics
    )

    print(
        "\n[PASS] train metrics are finite"
    )

    # --------------------------------------------------------
    # Check parameters actually updated
    # --------------------------------------------------------

    assert model_changed(
        aligned_model,
        aligned_initial_state,
    )

    assert model_changed(
        control_model,
        control_initial_state,
    )

    print(
        "[PASS] aligned model parameters changed"
    )

    print(
        "[PASS] control model parameters changed"
    )

    # Target should NOT change.
    target_changed = model_changed(
        target_model,
        target_initial_state,
    )

    assert not target_changed

    print(
        "[PASS] target model parameters did not change"
    )

    # They started identical, but should normally diverge because
    # aligned receives an additional CKA gradient.
    still_identical = models_are_identical(
        aligned_model,
        control_model,
    )

    print(
        "[INFO] aligned/control still identical after training:",
        still_identical,
    )

    # ========================================================
    # VALIDATION EPOCH
    # ========================================================

    print("\n" + "-" * 70)
    print("Running VALIDATION epoch")
    print("-" * 70)

    val_metrics = ta.run_epoch(
        split="val",
        epoch=0,
        loader=val_loader,
        target_model=target_model,
        aligned_model=aligned_model,
        control_model=control_model,
        aligned_optimizer=None,
        control_optimizer=None,
        cfg=cfg,
        device=device,
        logger=logger,
        sample_order_path=None,
    )

    print("\nValidation metrics:")
    print(val_metrics)

    assert_finite_metrics(
        val_metrics
    )

    print(
        "\n[PASS] validation metrics are finite"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("SANITY CHECK PASSED")
    print("=" * 70)

    print(
        """
Successfully tested:

✓ dummy DataLoader
✓ batch unpacking
✓ temporal input expansion
✓ frozen target forward pass
✓ pseudo-label generation
✓ target representation extraction
✓ aligned forward pass
✓ control forward pass
✓ cross-entropy loss
✓ CKA computation
✓ alignment loss
✓ aligned backward pass
✓ control backward pass
✓ optimizer updates
✓ validation mode
✓ metric aggregation
✓ CPU execution
"""
    )


if __name__ == "__main__":
    main()