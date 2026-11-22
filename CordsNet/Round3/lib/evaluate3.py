"""
Round 3 evaluation (epoch 0 and after every training epoch).

1. CKA and the ImageNet head on THINGS images: Round1/core/evaluate.py, protocol unchanged
   (val, one day per batch, batch 64, seed 2723, full batches only, all layers x areas x 20 bins,
   three estimators, pooled), with only the ImageNet head's logits (0-999) for the pseudo-label CE
   and the top-5. So CKA, agreement and THINGS top-1 / top-5 on overlapping concepts are computed
   exactly as in Rounds 1-2.
2. THINGS head: every val image of a non-overlapping concept (1306), batches of 64 without the
   day constraint (classification only): top-1 / top-5 at every stimulus step and the weighted CE
   (steps 70-99), as in Round3/things_head/train_head.py. At epoch 0 this reproduces the trained
   head's numbers (val top-1 0.630, top-5 0.835, CE 1.696).
"""
import numpy as np
import torch

from core.accuracy import accuracy
from core.evaluate import evaluate
from core.losses import weighted_ce
from core.recording import N_BINS, BinRecorder
from lib.model import IMAGENET, THINGS


@torch.no_grad()
def things_head_eval(model, cache, things_lab, splits=("val",), batch_size=64, n_steps=100, ce_steps=(70, 100)):
    rows = np.where(np.isin(cache.split, list(splits)) & (things_lab.cpu().numpy() >= 0))[0]
    top5 = np.zeros((len(rows), n_steps, 5), dtype=np.int16)
    ce_sum = 0.0
    was_training = model.training
    model.eval()
    rec = BinRecorder(model, [], steps_per_bin=n_steps // N_BINS)   # records nothing; keeps DynVision from storing responses
    try:
        for i in range(0, len(rows), batch_size):
            r = torch.as_tensor(rows[i:i + batch_size], device=cache.device)
            rec.reset()
            logits = model(cache.images_for(r).unsqueeze(1).expand(-1, n_steps, -1, -1, -1))[..., THINGS]
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


def evaluate_round3(model, cache, things_lab, ev, n_steps, ce_steps, overlap, log=print):
    """Returns (arrays to save, dict of summary numbers)."""
    r = evaluate(model, cache, ev["layers"], ev["areas"], tuple(ev["splits"]), ev["batch_size"], ev["seed"],
                 n_steps, ce_steps, log=log, out_slice=IMAGENET)
    acc = accuracy(r["top5"], r["eval_rows"], cache.pseudo_label.cpu().numpy(), cache.concept, overlap)
    th = things_head_eval(model, cache, things_lab, tuple(ev["splits"]), ev["batch_size"], n_steps, ce_steps)
    arrays = {**r, **{f"things_head_{k}": v for k, v in th.items()}}
    summary = {"val_ce_pseudo": r["ce_weighted"], "agree": acc["agree"],
               "imnet_head_things_top1": acc["things_top1"], "imnet_head_things_top5": acc["things_top5"],
               "things_head_top1": float(th["top1_t"][-1]), "things_head_top5": float(th["top5_t"][-1]),
               "things_head_ce": th["ce"], "things_head_n": len(th["rows"])}
    return arrays, summary
