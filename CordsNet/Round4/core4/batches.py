"""
Round 4 THINGS batches: one recording day per batch, every image used, every batch full.

Per day: the day's images in random order, cut into batches; if the last batch is not full, it is
filled with images drawn at random (without replacement) from the same day's other batches, so a
few images appear twice in the epoch but never twice in one batch. A day with fewer images than
the batch size gives one smaller batch; a day with fewer than MIN_SIZE (4) images is skipped (the
unbiased CKA_S needs at least 4). Rounds 1-3 instead dropped the partial batch.

  train_batches  new order every epoch, deterministic in (seed, epoch); the batches of all days are
                 shuffled together. The number of batches per epoch is the same every epoch.
  eval_batches   fixed for a given seed (same every evaluation), days in ascending order.

Same seed -> same batches in every run (control and students see the same images in the same order).
"""
import numpy as np

MIN_SIZE = 4


def day_batches(rows, batch_size, rng):
    """Batches (arrays of row indices) covering every row of one day; see the module docstring."""
    if len(rows) < MIN_SIZE:
        return []
    rows = rng.permutation(rows)
    if len(rows) <= batch_size:
        return [rows]
    n_full = len(rows) // batch_size
    out = [rows[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
    rest = rows[n_full * batch_size:]
    if len(rest):
        fill = rng.choice(rows[:n_full * batch_size], batch_size - len(rest), replace=False)
        out.append(np.concatenate([rest, fill]))
    return out


def _days(cache, splits):
    return [(d, np.where((cache.day == d) & np.isin(cache.split, splits))[0]) for d in sorted(set(cache.day.tolist()))]


def train_batches(cache, split, batch_size, seed, epoch):
    """List of (day, row_indices) for one training epoch."""
    rng = np.random.default_rng([seed, epoch])
    out = [(d, b) for d, rows in _days(cache, [split]) if len(rows) for b in day_batches(rows, batch_size, rng)]
    return [out[i] for i in rng.permutation(len(out))]


def eval_batches(cache, splits, batch_size, seed):
    """List of (day, row_indices): fixed evaluation batches (days ascending)."""
    rng = np.random.default_rng(seed)
    return [(d, b) for d, rows in _days(cache, list(splits)) if len(rows) for b in day_batches(rows, batch_size, rng)]
