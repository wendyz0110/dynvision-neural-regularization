"""
CordsNet-R8 with two parallel classification heads: ImageNet (1000) and THINGS (1854 concepts).

The backbone and the ImageNet head come from the CordsNet authors' checkpoint (cordsnetr8.pth,
loaded by common/cordsnet_model.py with every tensor verified). The THINGS head is a trained
Linear(512, 1854) on the same 512-d classifier input (layer7 + layer8 -> global average pool),
stored in one self-contained file (see "THINGS head file" below).

DynVision's forward calls the classifier once per time step and returns one tensor, so both heads
live in one weight_norm(Linear(512, 1000 + 1854)): outputs 0-999 are the ImageNet head (unchanged),
outputs 1000-2853 the THINGS head (THINGS class k = output 1000 + k). Each output depends only on
its own weight row, and weight_norm works per row, so this is exactly two separate heads; take each
head's slice of the logits (heads.imagenet, heads.things) and apply its own softmax / CE. Only the
last layer is replaced, so everything else in DynVision (time steps, idle period, response
recording) works as for the original model.

Usage:
    from common.cordsnet_things import build_cordsnet_things
    model, heads, info = build_cordsnet_things("cordsnetr8.pth", "things_head_1854.pt", trainable=True)
    logits = model(x)                               # x: (B, T, 3, 224, 224) -> logits (B, T, 2854)
    imagenet_logits = logits[..., heads.imagenet]   # (B, T, 1000), CordsNet's ImageNet class order
    things_logits = logits[..., heads.things]       # (B, T, 1854), class order = heads.things_classes
    y = heads.labels(concept_names)                 # THINGS concept names (uniqueID) -> class indices

THINGS head file (a dict saved with torch.save; loads with weights_only=True):
    format              "cordsnet_things_head_v1"
    state_dict          weight-norm Linear(512, 1854): parametrizations.weight.original0 (g, 1854 x 1),
                        parametrizations.weight.original1 (v, 1854 x 512), bias (1854)
    classes             1854 THINGS concept names (uniqueID), index = THINGS class
    is_overlap          1854 bools: concept matched to ImageNet classes
    imagenet_classes    1854 lists: the matched ImageNet class indices (empty if no overlap)
    description, training   how the head was made and its validation results
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.cordsnet_model import build_cordsnet

N_IMAGENET = 1000
HEAD_FORMAT = "cordsnet_things_head_v1"


@dataclass
class Heads:
    """Where each head sits in the model's output, and the THINGS class order."""
    imagenet: slice
    things: slice
    n_outputs: int
    things_classes: list
    things_is_overlap: np.ndarray
    things_imagenet_classes: list
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {c: i for i, c in enumerate(self.things_classes)}

    def labels(self, concepts, device=None):
        """THINGS class index for each concept name (LongTensor). Unknown names raise an error."""
        concepts = [str(c) for c in concepts]
        unknown = sorted(set(c for c in concepts if c not in self._index))
        if unknown:
            raise KeyError(f"{len(unknown)} concepts not in the THINGS head, e.g. {unknown[:5]}")
        return torch.tensor([self._index[c] for c in concepts], dtype=torch.long, device=device)


def load_things_head(path):
    """The THINGS head file as a dict (see the module docstring), with shapes checked."""
    head = torch.load(path, map_location="cpu", weights_only=True)
    assert head.get("format") == HEAD_FORMAT, f"{path}: not a {HEAD_FORMAT} file"
    sd, n = head["state_dict"], len(head["classes"])
    g, v, b = sd["parametrizations.weight.original0"], sd["parametrizations.weight.original1"], sd["bias"]
    assert g.shape == (n, 1) and v.shape == (n, 512) and b.shape == (n,), \
        f"unexpected THINGS head shapes {tuple(g.shape)} {tuple(v.shape)} {tuple(b.shape)} for {n} classes"
    assert len(head["is_overlap"]) == n and len(head["imagenet_classes"]) == n
    return head


def build_cordsnet_things(checkpoint_path, things_head_path, trainable=True, **model_kwargs):
    """Pretrained CordsNet-R8 whose classifier layer is weight_norm(Linear(512, 1000 + n_things))
    = [ImageNet head (from the checkpoint); THINGS head (from things_head_path)].

    trainable: True -> every parameter trainable except CordsNet's fixed time constants;
               False -> all frozen (as common/cordsnet_model.build_cordsnet).
    model_kwargs: CordsNet settings (n_timesteps, idle_timesteps, dt, tau, ...), defaults as in
               common/cordsnet_model.DEFAULT_MODEL_KWARGS.

    Returns (model, heads, info).
    """
    model, info = build_cordsnet(checkpoint_path, trainable=trainable, n_classes=N_IMAGENET, **model_kwargs)
    head = load_things_head(things_head_path)
    n_things = len(head["classes"])
    n_out = N_IMAGENET + n_things
    old = model.classifier[3]
    assert isinstance(old, torch.nn.Linear) and old.in_features == 512 and old.out_features == N_IMAGENET, \
        f"unexpected CordsNet classifier layer: {old}"

    new = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(512, n_out, bias=True))
    sd = head["state_dict"]
    with torch.no_grad():
        p_old = old.parametrizations.weight
        new.parametrizations.weight.original0.copy_(torch.cat([p_old.original0.detach().cpu(), sd["parametrizations.weight.original0"]], 0))
        new.parametrizations.weight.original1.copy_(torch.cat([p_old.original1.detach().cpu(), sd["parametrizations.weight.original1"]], 0))
        new.bias.copy_(torch.cat([old.bias.detach().cpu(), sd["bias"]], 0))
        # check: the two parts are exactly the original ImageNet head and the stored THINGS head
        assert torch.equal(new.weight[:N_IMAGENET], old.weight.detach().cpu()), "ImageNet rows changed"
        assert torch.equal(new.bias[:N_IMAGENET], old.bias.detach().cpu()), "ImageNet bias changed"
        g, v = sd["parametrizations.weight.original0"], sd["parametrizations.weight.original1"]
        assert torch.allclose(new.weight[N_IMAGENET:], g * v / v.norm(dim=1, keepdim=True), atol=1e-6), \
            "THINGS rows differ from the stored head"

    model.classifier[3] = new
    model.n_classes = n_out
    if trainable:
        model.trainable_parameter_names = list(model.state_dict().keys())
    else:
        for p in new.parameters():
            p.requires_grad = False

    heads = Heads(imagenet=slice(0, N_IMAGENET), things=slice(N_IMAGENET, n_out), n_outputs=n_out,
                  things_classes=[str(c) for c in head["classes"]],
                  things_is_overlap=np.array(head["is_overlap"], dtype=bool),
                  things_imagenet_classes=[list(c) for c in head["imagenet_classes"]])
    info = {**info, "n_classes": n_out,
            "heads": {"imagenet": [0, N_IMAGENET], "things": [N_IMAGENET, n_out]},
            "things_head": {"path": str(Path(things_head_path).resolve()), "n_classes": n_things,
                            "description": head.get("description", "")}}
    return model, heads, info
