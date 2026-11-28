"""
Round 4 evaluation (epoch 0 and after every training epoch).

1. CKA: Round1/core/evaluate.py (all layers x areas x 20 bins, three estimators, pooled and
   per-batch mean, batch 64, one day per batch), on the Round 4 eval batches (core4/batches.py):
   every val image is used; a day's last batch is filled with other images of that day.
   Only the ImageNet head's logits are passed on for its pseudo-label CE / top-5 (kept in the
   arrays, not summarized).
2. THINGS head (all 1854 concepts): every val image once, batches of 64 without the day
   constraint (classification only): top-1 / top-5 at every stimulus step, weighted CE (steps 70-99).
"""
import numpy as np
import torch

from core.evaluate import evaluate
from core.losses import weighted_ce
from core.recording import N_BINS, BinRecorder
from core4.batches import eval_batches


@torch.no_grad()
def things_eval(model, cache, things_lab, things_slice, splits=("val",), batch_size=64, n_steps=100, ce_steps=(70, 100)):
    rows = np.where(np.isin(cache.split, list(splits)))[0]
    top5 = np.zeros((len(rows), n_steps, 5), dtype=np.int16)
    ce_sum = 0.0
    was_training = model.training
    model.eval()
    rec = BinRecorder(model, [], steps_per_bin=n_steps // N_BINS)   # records nothing; keeps DynVision from storing responses
    try:
        for i in range(0, len(rows), batch_size):
            r = torch.as_tensor(rows[i:i + batch_size], device=cache.device)
            rec.reset()
            logits = model(cache.images_for(r).unsqueeze(1).expand(-1, n_steps, -1, -1, -1))[..., things_slice]
            top5[i:i + len(r)] = logits.float().topk(5, dim=-1).indices.cpu().numpy()
            ce_sum += float(weighted_ce(logits, things_lab[r], *ce_steps)) * len(r)
    finally:
        rec.detach()
        model.train(was_training)
    lab = things_lab.cpu().numpy()[rows]
    hit1 = top5[:, :, 0] == lab[:, None]
    hit5 = (top5 == lab[:, None, None]).any(-1)
    return {"rows": rows, "top5": top5, "label": lab, "top1_t": hit1.mean(0), "top5_t": hit5.mean(0),
            "ce": ce_sum / len(rows)}


def evaluate_round4(model, heads, cache, things_lab, ev, n_steps, ce_steps, log=print):
    """Returns (arrays to save, dict of summary numbers)."""
    batches = eval_batches(cache, ev["splits"], ev["batch_size"], ev["seed"])
    r = evaluate(model, cache, ev["layers"], ev["areas"], tuple(ev["splits"]), ev["batch_size"], ev["seed"],
                 n_steps, ce_steps, log=log, out_slice=heads.imagenet, batches=batches)
    th = things_eval(model, cache, things_lab, heads.things, tuple(ev["splits"]), ev["batch_size"], n_steps, ce_steps)
    arrays = {**r, "batch_size": np.array([len(b) for _, b in batches]),
              **{f"things_head_{k}": v for k, v in th.items()}}
    summary = {"things_top1": float(th["top1_t"][-1]), "things_top5": float(th["top5_t"][-1]),
               "things_ce": th["ce"], "things_n": len(th["rows"]), "cka_batches": len(batches)}
    return arrays, summary
