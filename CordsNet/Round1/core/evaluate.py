"""
Evaluation on fixed batches, the same protocol as epoch 0 (Round1/epoch0/run_epoch0.py):
images of the eval split, one day per batch, batch size 64, epoch-0 shuffle seed, full
batches only, float32, all layers x areas x 20 bins, three CKA estimators. Also the CE
against the pseudo-labels at every stimulus step, and the top-5 ImageNet classes of every
evaluated image at every step (for accuracy: agreement with the pseudo-labels, THINGS labels).

Used for epoch 0 of every training run and after every training epoch, so all numbers are
directly comparable with Round1/outputs/epoch0_pretrained.
"""
import warnings

import numpy as np
import torch

from cka_estimators import cka_all, h_cross, h_self, moments
from core.data import eval_batches
from core.losses import ce_per_step, ce_weights
from core.recording import N_BINS, BinRecorder

ESTIMATORS = ["CKA_0", "CKA_S", "CKA_C"]


@torch.no_grad()
def evaluate(model, cache, layers, areas, splits=("val",), batch_size=64, seed=2723, n_steps=100,
             ce_steps=(70, 100), log=print, out_slice=None, batches=None):
    """out_slice: use only these model outputs for the pseudo-label CE and the top-5 (Round 3:
    the ImageNet head, logits 0-999, of the two-head model). None = all outputs (Rounds 1-2).
    batches: list of (day, row_indices) to evaluate instead of eval_batches(...) (Round 4: every
    val image, last batch of a day filled; batches may differ in size). None = Rounds 1-3 protocol."""
    if batches is None:
        batches = eval_batches(cache, splits, batch_size, seed)
    nb, nl, na = len(batches), len(layers), len(areas)
    h_num = np.full((nb, nl, na, N_BINS, 2), np.nan)
    h_den_model = np.full((nb, nl, N_BINS, 3), np.nan)
    h_den_neural = np.full((nb, na, N_BINS, 3), np.nan)
    cka_batch = np.full((nb, nl, na, N_BINS, 3), np.nan)
    timecourse = np.zeros((nl, N_BINS))
    ce_t = np.zeros(n_steps)
    eval_rows = np.concatenate([rows for _, rows in batches]) if batches else np.zeros(0, dtype=int)
    top5 = np.zeros((len(eval_rows), n_steps, 5), dtype=np.int16)     # (image, step, rank), order of eval_rows
    start = np.cumsum([0] + [len(rows) for _, rows in batches])

    was_training = model.training
    model.eval()
    rec = BinRecorder(model, layers, steps_per_bin=n_steps // N_BINS)
    try:
        for b, (day, rows) in enumerate(batches):
            rows_t = torch.as_tensor(rows, device=cache.device)
            rec.reset()
            x = cache.images_for(rows_t)
            logits = model(x.unsqueeze(1).expand(-1, n_steps, -1, -1, -1))
            if out_slice is not None:
                logits = logits[..., out_slice]
            rec.check_complete(n_steps)
            ce_t += ce_per_step(logits, cache.pseudo_label[rows_t]).cpu().numpy() / nb
            top5[start[b]:start[b + 1]] = logits.float().topk(5, dim=-1).indices.cpu().numpy()

            for k in range(N_BINS):
                hy = {}
                for j, area in enumerate(areas):
                    my = moments(cache.neural[area][rows_t, k])
                    hy[area] = (my, h_self(my))
                    h_den_neural[b, j, k] = [float(v) for v in hy[area][1]]
                for i, name in enumerate(layers):
                    a = rec.bins[name][k]
                    mx = moments(a)
                    hx = h_self(mx)
                    h_den_model[b, i, k] = [float(v) for v in hx]
                    timecourse[i, k] += float(a.mean()) / nb
                    for j, area in enumerate(areas):
                        my, hyy = hy[area]
                        h_num[b, i, j, k] = [float(v) for v in h_cross(mx, my)]
                        cka_batch[b, i, j, k] = [float(v) for v in cka_all(mx, my, hx, hyy)]
            if (b + 1) % 10 == 0 or b + 1 == nb:
                log(f"    eval batch {b + 1}/{nb}")
    finally:
        rec.detach()
        model.train(was_training)

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cka_mean = np.nanmean(cka_batch, axis=0)
        undefined = np.isnan(cka_batch).all(axis=0)
        num, dm, dn = h_num.mean(0), h_den_model.mean(0), h_den_neural.mean(0)
        cka_pooled = np.stack([
            num[..., 0] / np.sqrt(dm[:, None, :, 0] * dn[None, :, :, 0]),
            num[..., 1] / np.sqrt(dm[:, None, :, 1] * dn[None, :, :, 1]),
            num[..., 1] / np.sqrt(dm[:, None, :, 2] * dn[None, :, :, 2]),
        ], axis=-1)
        cka_pooled[undefined] = np.nan

    w = ce_weights(ce_steps[1] - ce_steps[0]).numpy()
    return {
        "cka_batch": cka_batch, "cka_mean": cka_mean, "cka_pooled": cka_pooled,
        "h_num": h_num, "h_den_model": h_den_model, "h_den_neural": h_den_neural,
        "model_timecourse": timecourse, "batch_day": np.array([d for d, _ in batches]),
        "ce_per_step": ce_t, "ce_weighted": float((w * ce_t[ce_steps[0]:ce_steps[1]]).sum()),
        "top5": top5, "eval_rows": eval_rows,
        "layers": np.array(layers), "areas": np.array(areas), "estimators": np.array(ESTIMATORS),
        "bin_start_ms": np.arange(N_BINS) * 10,
    }
