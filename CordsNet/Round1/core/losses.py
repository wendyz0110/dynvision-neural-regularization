"""
Round 1 losses.

CE (classification, original CordsNet form): cross-entropy against the pseudo-label at the
last 30 stimulus steps (outputs 70-99 = 142-200 ms), log-weighted with logspace(-3, 0, 30),
combined as a weighted MEAN (sum of weights ~4.7 divided out), so its scale is that of one
step's CE, like the plain time-averaged CE of rounds 3/4.

CKA (alignment): for each selected bin k, CKA_S (Song et al. 2012) between the model layer's
bin-mean activity and the neural data of the same images and bin, then
    L_CKA = mean over valid bins of log(max(1 - CKA_S, floor)).
A bin is skipped (NaN, no gradient) when
  - undefined: the layer gives (almost) the same activity for every image in the batch
    (e.g. a deep layer before the image information has arrived), or
  - failed: a Song denominator is <= 0 (possible estimation noise with small batches).
CKA is computed in float64 outside autocast.

Several alignment targets (Round 2): one (area, layer, bins, weight) per target,
    loss = w_ce * CE + sum over targets of weight * L_CKA(target)
(`parse_targets`; the Round 1 single-target config keys still work).
"""
import torch
import torch.nn.functional as F

from cka_estimators import h_cross, h_self, moments


def ce_weights(n_steps=30, device=None):
    w = torch.logspace(-3, 0, n_steps, device=device)
    return w / w.sum()


def weighted_ce(logits, labels, first_step=70, last_step=100):
    """logits (B, T, C); labels (B,). Weighted mean CE over steps first_step..last_step-1."""
    x = logits[:, first_step:last_step].float()                      # (B, 30, C)
    n = x.shape[1]
    ce_t = F.cross_entropy(x.reshape(-1, x.shape[-1]), labels.repeat_interleave(n), reduction="none")
    ce_t = ce_t.view(-1, n).mean(0)                                    # (30,) mean over the batch
    return (ce_weights(n, x.device) * ce_t).sum()


def ce_per_step(logits, labels):
    """CE at every stimulus step, mean over the batch: (T,)."""
    x = logits.float()
    T = x.shape[1]
    ce = F.cross_entropy(x.reshape(-1, x.shape[-1]), labels.repeat_interleave(T), reduction="none")
    return ce.view(-1, T).mean(0)


def cka_s_bins(model_bins, neural, min_rel_var=1e-8):
    """CKA_S per bin, keeping the graph.

    model_bins: list of 20 tensors (B, D) (None allowed); neural: (B, 20, E).
    Returns (list of 20 values: tensor or None, status list: "ok" / "undefined" / "failed").
    """
    vals, status = [], []
    with torch.autocast(device_type=neural.device.type, enabled=False):
        for k, X in enumerate(model_bins):
            if X is None:
                vals.append(None), status.append("undefined")
                continue
            mx, my = moments(X), moments(neural[:, k])
            if min(mx["rel_var"], my["rel_var"]) < min_rel_var:
                vals.append(None), status.append("undefined")
                continue
            hx, hy = h_self(mx)[1], h_self(my)[1]
            if hx.item() <= 0 or hy.item() <= 0:
                vals.append(None), status.append("failed")
                continue
            vals.append(h_cross(mx, my)[1] / torch.sqrt(hx * hy))
            status.append("ok")
    return vals, status


def cka_loss(vals, loss_bins, floor=1e-6):
    """mean over the valid selected bins of log(max(1 - CKA_S, floor)); None if no valid bin."""
    terms = [torch.log(torch.clamp(1 - vals[k], min=floor)) for k in loss_bins if vals[k] is not None]
    return torch.stack(terms).mean() if terms else None


def as_float(v):
    return float("nan") if v is None else float(v.detach())


def parse_targets(loss_cfg):
    """Alignment targets from the loss config, as a list of dicts
    {area, layer, bins (list of bin indices), weight, tag ("<area>_<layer>")}.

    Round 2 form:   targets: [{area: V1, layer: layer2, bins: [2, 19], weight: 1.0}, ...]
    Round 1 form:   area, layer, bins, w_cka  -> one target with weight w_cka
    bins are [first, last], inclusive.
    """
    raw = loss_cfg.get("targets")
    if raw is None:
        raw = [{"area": loss_cfg["area"], "layer": loss_cfg["layer"], "bins": loss_cfg["bins"], "weight": loss_cfg["w_cka"]}]
    targets = []
    for t in raw:
        targets.append({"area": t["area"], "layer": t["layer"], "bins": list(range(t["bins"][0], t["bins"][1] + 1)),
                        "weight": float(t["weight"]), "tag": f"{t['area']}_{t['layer']}"})
    tags = [t["tag"] for t in targets]
    assert len(set(tags)) == len(tags), f"duplicate alignment targets: {tags}"
    return targets
