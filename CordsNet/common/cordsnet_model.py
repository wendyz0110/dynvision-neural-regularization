"""
Load DynVision's CordsNet with the original authors' pretrained checkpoint.

DynVision's CordsNet re-implements the architecture of wmws2/cordsnet and, when
`init_with_pretrained=True`, loads that repo's `cordsnetr8.pth` (ImageNet-1k)
by translating parameter names. Its built-in loader downloads the file to a
hard-coded path that does not exist on our machines, so here we subclass
CordsNet and hand it the state dict from a local file instead. The name
translation and loading are still done by DynVision itself.

Shared by target (frozen) and, later, student (trainable) models.
"""
from pathlib import Path

import torch
from dynvision.models import CordsNet

# Dynamics settings matching the original paper / repo:
# tau = 10 ms, dt = 2 ms (alpha = dt / tau = 0.2), 100 idle steps (200 ms) of
# spontaneous activity, then 100 steps (200 ms) of stimulus presentation.
DEFAULT_MODEL_KWARGS = dict(
    n_timesteps=100,
    idle_timesteps=100,
    dt=2.0,
    tau=10.0,
    t_feedforward=2.0,
    t_recurrence=2.0,
    t_skip=2.0,
)


def _translate_key(key, translation):
    """Map an original-repo parameter name to DynVision's name (same rule as DynVision)."""
    for old, new in translation.items():
        if old in key:
            return key.replace(old, new)
    return key


def verify_pretrained_loaded(model, checkpoint_state_dict):
    """Check that every checkpoint tensor ended up in the model unchanged.

    DynVision loads with strict=False, so a renamed or missing layer would be
    silently left at its random initialization. Raise instead.
    """
    translation = model.translate_pretrained_layer_names()
    model_sd = model.state_dict()
    problems = []
    for key, value in checkpoint_state_dict.items():
        new_key = _translate_key(key, translation)
        if new_key not in model_sd:
            problems.append(f"missing in model: {key} -> {new_key}")
        elif model_sd[new_key].shape != value.shape:
            problems.append(f"shape mismatch: {key} {tuple(value.shape)} vs {tuple(model_sd[new_key].shape)}")
        elif not torch.equal(model_sd[new_key].detach().cpu(), value.cpu()):
            problems.append(f"value mismatch: {key} -> {new_key}")
    if problems:
        raise RuntimeError(
            "Pretrained CordsNet weights were not loaded correctly:\n  " + "\n  ".join(problems)
        )
    return len(checkpoint_state_dict)


def build_cordsnet(checkpoint_path, trainable=False, n_classes=1000, image_size=224, **model_kwargs):
    """Build DynVision's CordsNet initialized from a local CordsNet checkpoint.

    Args:
        checkpoint_path: path to the original repo's `cordsnetr8.pth`.
        trainable: False -> all parameters frozen (target / inference).
                   True  -> all parameters trainable except the fixed time
                   constants (student). Note DynVision's own default with a
                   pretrained init is to train only the classifier; we override it.
        n_classes: number of output classes (1000 = keep the ImageNet readout).
        image_size: input height/width (CordsNet's spatial biases require 224).
        **model_kwargs: overrides for DEFAULT_MODEL_KWARGS.

    Returns:
        (model, info) where info records the settings used.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    kwargs = {**DEFAULT_MODEL_KWARGS, **model_kwargs}

    def _load_local(self):
        # DynVision may modify the dict it receives, so give it a copy.
        return {k: v.clone() for k, v in ckpt.items()}

    LocalCordsNet = type("LocalCordsNet", (CordsNet,), {"download_pretrained_state_dict": _load_local})
    model = LocalCordsNet(
        input_dims=(kwargs["n_timesteps"], 3, image_size, image_size),
        n_classes=n_classes,
        init_with_pretrained=True,
        **kwargs,
    )

    n_checked = verify_pretrained_loaded(model, ckpt)

    if trainable:
        # requires_grad is left as constructed (time constants stay fixed);
        # make DynVision treat every parameter as trainable.
        model.trainable_parameter_names = list(model.state_dict().keys())
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.trainable_parameter_names = []
        model.eval()

    info = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_tensors_verified": n_checked,
        "trainable": trainable,
        "n_classes": n_classes,
        "image_size": image_size,
        "model_kwargs": kwargs,
    }
    return model, info


def repeat_over_time(images, n_timesteps):
    """(B, C, H, W) -> (B, T, C, H, W): show the same image at every timestep."""
    return images.unsqueeze(1).expand(-1, n_timesteps, -1, -1, -1)
