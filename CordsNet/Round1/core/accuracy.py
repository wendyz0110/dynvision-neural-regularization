"""
Classification accuracy from the evaluation's top-5 predictions (core/evaluate.py):
  agreement   prediction = pseudo-label (the pretrained model's last-step prediction)
  THINGS top-1 / top-5   only images of THINGS concepts matched to ImageNet classes
              (prepare_images/things_imagenet_overlap.csv); correct if the prediction is one
              of the concept's matched classes
All at one stimulus step (default: the last, 200 ms).
"""
import csv

import numpy as np


def load_overlap(path):
    """{THINGS concept: [ImageNet class indices]} from things_imagenet_overlap.csv."""
    with open(path, newline="") as fh:
        return {r["uniqueID"]: [int(i) for i in r["imagenet_idx"].split(";")] for r in csv.DictReader(fh)}


def accuracy(top5, rows, pseudo_label, concept, overlap=None, step=-1):
    """top5 (images, steps, 5); rows: cache rows of those images; pseudo_label, concept: per cache row.
    Returns {"agree": ..., "things_top1": ..., "things_top5": ..., "things_n": ...} (THINGS NaN without overlap)."""
    pred = top5[:, step].astype(int)                                     # (images, 5)
    out = {"agree": float((pred[:, 0] == pseudo_label[rows]).mean()) if len(rows) else float("nan")}
    hits1, hits5 = [], []
    for k, r in enumerate(rows):
        classes = (overlap or {}).get(str(concept[r]))
        if classes:
            hits1.append(pred[k, 0] in classes)
            hits5.append(any(c in classes for c in pred[k]))
    out["things_n"] = len(hits1)
    out["things_top1"] = float(np.mean(hits1)) if hits1 else float("nan")
    out["things_top5"] = float(np.mean(hits5)) if hits5 else float("nan")
    return out
