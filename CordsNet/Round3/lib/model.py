"""
CordsNet-R8 with two parallel classification heads (Round 3).

CordsNet's readout: layer7 + layer8 -> global average pool -> 512 values ->
weight_norm(Linear(512, 1000)) (model.classifier[3]). Round 3 adds a THINGS head for the
1306 THINGS concepts that do not overlap ImageNet (trained on the frozen features, see
Round3/things_head/train_head.py), on the same 512-d input.

DynVision's forward calls the classifier once per time step and returns one tensor, so both
heads live in one weight_norm(Linear(512, 1000 + 1306)): rows 0-999 are the ImageNet head,
rows 1000-2305 the THINGS head (THINGS class k = logit 1000 + k). Each output is set by its
own weight row only, and weight_norm works per row (g, v per output), so this is exactly two
separate heads; each CE takes its own slice of the logits (IMAGENET, THINGS), with its own
softmax.
"""
import sys
from pathlib import Path

import numpy as np
import torch

CORDSNET_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORDSNET_DIR))

from common.cordsnet_model import build_cordsnet  # noqa: E402

N_IMAGENET, N_THINGS = 1000, 1306
IMAGENET = slice(0, N_IMAGENET)
THINGS = slice(N_IMAGENET, N_IMAGENET + N_THINGS)
N_OUT = N_IMAGENET + N_THINGS


def load_things_head(head_path, results_path):
    """The trained THINGS head (weight-norm state dict) and its class order (concept names)."""
    sd = torch.load(head_path, map_location="cpu", weights_only=True)
    classes = [str(c) for c in np.load(results_path)["classes"]]
    g, v, b = sd["parametrizations.weight.original0"], sd["parametrizations.weight.original1"], sd["bias"]
    assert g.shape == (len(classes), 1) and v.shape == (len(classes), 512) and b.shape == (len(classes),), \
        f"unexpected THINGS head shapes {tuple(g.shape)} {tuple(v.shape)} {tuple(b.shape)}"
    return {"g": g, "v": v, "b": b}, classes


def build_two_head_model(checkpoint_path, things_head_path, things_results_path, trainable=True, **model_kwargs):
    """Pretrained CordsNet-R8 whose classifier layer is replaced by
    weight_norm(Linear(512, 1000 + 1306)) = [ImageNet head; THINGS head].

    Returns (model, info, classes): classes = THINGS concept names, index = THINGS class.
    """
    model, info = build_cordsnet(checkpoint_path, trainable=trainable, n_classes=N_IMAGENET, **model_kwargs)
    old = model.classifier[3]
    head, classes = load_things_head(things_head_path, things_results_path)
    assert len(classes) == N_THINGS, f"THINGS head has {len(classes)} classes, expected {N_THINGS}"

    new = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(512, N_OUT, bias=True))
    with torch.no_grad():
        p_old = old.parametrizations.weight
        new.parametrizations.weight.original0.copy_(torch.cat([p_old.original0.detach().cpu(), head["g"]], 0))
        new.parametrizations.weight.original1.copy_(torch.cat([p_old.original1.detach().cpu(), head["v"]], 0))
        new.bias.copy_(torch.cat([old.bias.detach().cpu(), head["b"]], 0))
        # check: the two halves are exactly the old ImageNet head and the trained THINGS head
        assert torch.equal(new.weight[IMAGENET], old.weight.detach().cpu()), "ImageNet rows changed"
        assert torch.equal(new.bias[IMAGENET], old.bias.detach().cpu()), "ImageNet bias changed"
        w_things = head["g"] * head["v"] / head["v"].norm(dim=1, keepdim=True)
        assert torch.allclose(new.weight[THINGS], w_things, atol=1e-6), "THINGS rows differ from the trained head"

    model.classifier[3] = new
    model.n_classes = N_OUT
    if trainable:
        model.trainable_parameter_names = list(model.state_dict().keys())
    else:
        for p in new.parameters():
            p.requires_grad = False
    info = {**info, "n_classes": N_OUT, "heads": {"imagenet": [0, N_IMAGENET], "things": [N_IMAGENET, N_OUT]},
            "things_head": {"path": str(Path(things_head_path).resolve()), "classes": str(Path(things_results_path).resolve()),
                            "n_classes": N_THINGS}}
    return model, info, classes

