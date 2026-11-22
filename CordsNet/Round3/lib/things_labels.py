"""
THINGS-head labels for the cache rows.

The THINGS head covers the 1306 THINGS concepts that do not overlap ImageNet; its class order
is the `classes` array saved with the trained head (Round3/things_head, concept names sorted).
Images of the 548 overlapping concepts get label -1: they get no THINGS CE (they still enter the
CKA loss; their recognition is kept up by the ImageNet CE).
"""
import numpy as np
import torch


def things_labels(cache, classes):
    """LongTensor on the cache's device: THINGS class index per cache row, -1 = overlapping concept."""
    idx = {c: i for i, c in enumerate(classes)}
    lab = np.array([idx.get(str(c), -1) for c in cache.concept], dtype=np.int64)
    return torch.from_numpy(lab).to(cache.device)
