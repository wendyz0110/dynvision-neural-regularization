"""
CKA between the two monkeys (N vs F), same area (V1-V1, V4-V4, IT-IT): a reference for the
model-vs-monkey CKA (how similar two brains are on the same images).

Computed exactly like the model evaluation (core/evaluate.py), with "model layer" replaced by
the other monkey's area: val split, batches of 64, the three estimators of cka_estimators.py
per 10 ms bin, per-batch values and the pooled value (numerators and denominators averaged
over batches, then one ratio). Both monkeys saw the same images (same train/val/test split),
but on different recording days, so two batchings are computed:
  N_day   the model evaluation's batches (one monkey-N day per batch; identical rows), so the
          numbers are directly comparable with the model's; F's images in a batch can come
          from two F days
  pair    batches from one (N day, F day) pair: both monkeys from one day per batch
          (7 pairs in the val split; same shuffle rule as the evaluation, partial batches dropped)
Response windows as in training: V1 bins 2-19, V4 5-19, IT 7-19 (window value = mean over
the window's bins of the pooled CKA, as in the Round 2 notebook).

Only the neural data are read (meta.npz + neural.npz of the two caches); no images, no GPU.

Output: <out_dir>/monkey_cka.npz, per batching <b> in (N_day, pair):
  <b>_cka_batch   (batches, areas, 20 bins, 3 estimators)
  <b>_cka_mean    (areas, 20, 3)   mean over batches
  <b>_cka_pooled  (areas, 20, 3)
  <b>_h_num (batches, areas, 20, 2), <b>_h_den_N / <b>_h_den_F (batches, areas, 20, 3)
  <b>_window_pooled (areas, 3)    mean over the area's window bins of cka_pooled
  <b>_batch_day_N, <b>_batch_day_F (batches,)  day of each batch (-1 = mixed days)
  areas, estimators, bin_start_ms, window_bins_<area>

Usage (on Torch, CPU job, see ../slurm/monkey_cka.sbatch):
  python monkey_cka.py
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROUND3_DIR = Path(__file__).resolve().parents[1]
CORDSNET_DIR = ROUND3_DIR.parent
REPO_ROOT = CORDSNET_DIR.parent
sys.path.insert(0, str(CORDSNET_DIR / "Round1"))

from cka_estimators import cka_all, h_cross, h_self, moments  # noqa: E402
from core.data import eval_batches  # noqa: E402
from core.recording import N_BINS  # noqa: E402

AREAS = ["V1", "V4", "IT"]
WINDOW = {"V1": np.arange(2, 20), "V4": np.arange(5, 20), "IT": np.arange(7, 20)}   # as in training
ESTIMATORS = ["CKA_0", "CKA_S", "CKA_C"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_neural(cache_dir):
    """meta (stim_idx, day, split) and neural data {area: (N, 20, E) float32} of one cache."""
    meta = np.load(Path(cache_dir) / "meta.npz")
    nz = np.load(Path(cache_dir) / "neural.npz")
    c = SimpleNamespace(stim_idx=meta["stim_idx"], day=meta["day"], split=meta["split"])
    neural = {a: torch.from_numpy(nz[a]) for a in AREAS}          # float32; moments() works in float64
    return c, neural


def pair_batches(day_n, day_f, rows_split, batch_size, seed):
    """Batches from one (N day, F day) pair: per pair (ascending), the rows sorted, shuffled with
    one generator shared across pairs, cut into full batches. Returns [(day_N, day_F, rows)]."""
    rng = np.random.default_rng(seed)
    out = []
    pairs = sorted({(int(day_n[r]), int(day_f[r])) for r in rows_split})
    for dn, df in pairs:
        rows = sorted(r for r in rows_split if day_n[r] == dn and day_f[r] == df)
        rows = rng.permutation(rows).tolist()
        for i in range(0, len(rows) - batch_size + 1, batch_size):
            out.append((dn, df, np.array(rows[i:i + batch_size])))
    return out


def monkey_cka(batches, neural_n, neural_f):
    """Per-batch moments and CKA between N and F, same area, every bin (as core/evaluate.py)."""
    nb, na = len(batches), len(AREAS)
    h_num = np.full((nb, na, N_BINS, 2), np.nan)
    h_den_n = np.full((nb, na, N_BINS, 3), np.nan)
    h_den_f = np.full((nb, na, N_BINS, 3), np.nan)
    cka_batch = np.full((nb, na, N_BINS, 3), np.nan)
    for b, rows in enumerate(batches):
        rows_t = torch.as_tensor(rows)
        for j, area in enumerate(AREAS):
            for k in range(N_BINS):
                mn, mf = moments(neural_n[area][rows_t, k]), moments(neural_f[area][rows_t, k])
                hn, hf = h_self(mn), h_self(mf)
                h_den_n[b, j, k] = [float(v) for v in hn]
                h_den_f[b, j, k] = [float(v) for v in hf]
                h_num[b, j, k] = [float(v) for v in h_cross(mn, mf)]
                cka_batch[b, j, k] = [float(v) for v in cka_all(mn, mf, hn, hf)]

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cka_mean = np.nanmean(cka_batch, axis=0)
        undefined = np.isnan(cka_batch).all(axis=0)
        num, dn, df = h_num.mean(0), h_den_n.mean(0), h_den_f.mean(0)
        cka_pooled = np.stack([
            num[..., 0] / np.sqrt(dn[..., 0] * df[..., 0]),
            num[..., 1] / np.sqrt(dn[..., 1] * df[..., 1]),
            num[..., 1] / np.sqrt(dn[..., 2] * df[..., 2]),
        ], axis=-1)
        cka_pooled[undefined] = np.nan
        window = np.stack([np.nanmean(cka_pooled[j, WINDOW[a]], axis=0) for j, a in enumerate(AREAS)])
    return {"cka_batch": cka_batch, "cka_mean": cka_mean, "cka_pooled": cka_pooled, "h_num": h_num,
            "h_den_N": h_den_n, "h_den_F": h_den_f, "window_pooled": window}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_N", default="/scratch/wz2723/data/TVSD/round1_cache_monkeyN")
    ap.add_argument("--cache_F", default="/scratch/wz2723/data/TVSD/round1_cache_monkeyF")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2723)      # the evaluation's shuffle seed
    ap.add_argument("--out_dir", default="CordsNet/Round3/outputs/monkey_cka")
    args = ap.parse_args()
    log = lambda s: print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)  # noqa: E731
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cn, neural_n = load_neural(args.cache_N)
    cf, neural_f = load_neural(args.cache_F)
    # row k = stim_idx k + 1 in both caches; same split, different days
    assert np.array_equal(cn.stim_idx, cf.stim_idx), "caches are not in the same stim_idx order"
    assert np.array_equal(cn.split, cf.split), "monkeys N and F have different splits"
    log("N: " + ", ".join(f"{a} {tuple(neural_n[a].shape)}" for a in AREAS))
    log("F: " + ", ".join(f"{a} {tuple(neural_f[a].shape)}" for a in AREAS))

    rows_split = np.where(cn.split == args.split)[0]
    batchings = {}
    # N_day: the model evaluation's batches (eval_batches gives row indices)
    ev = eval_batches(cn, [args.split], args.batch_size, args.seed)
    batchings["N_day"] = []
    for d, rows in ev:
        f_days = np.unique(cf.day[rows])
        batchings["N_day"].append((d, f_days[0] if len(f_days) == 1 else -1, rows))
    batchings["pair"] = pair_batches(cn.day, cf.day, rows_split, args.batch_size, args.seed)

    out = {"areas": np.array(AREAS), "estimators": np.array(ESTIMATORS),
           "bin_start_ms": np.arange(N_BINS) * 10, "split": args.split, "batch_size": args.batch_size}
    for a in AREAS:
        out[f"window_bins_{a}"] = WINDOW[a]
    for name, batches in batchings.items():
        n_img = sum(len(r) for _, _, r in batches)
        log(f"{name}: {len(batches)} batches, {n_img} of {len(rows_split)} {args.split} images")
        res = monkey_cka([r for _, _, r in batches], neural_n, neural_f)
        for k, v in res.items():
            out[f"{name}_{k}"] = v
        out[f"{name}_batch_day_N"] = np.array([int(dn) for dn, _, _ in batches])
        out[f"{name}_batch_day_F"] = np.array([int(df) for _, df, _ in batches])
        log(f"  {name} window CKA_S (pooled): " +
            ", ".join(f"{a} {res['window_pooled'][j, 1]:.3f}" for j, a in enumerate(AREAS)))

    path = out_dir / "monkey_cka.npz"
    np.savez(path, **out)
    log(f"saved {path}")


if __name__ == "__main__":
    main()
