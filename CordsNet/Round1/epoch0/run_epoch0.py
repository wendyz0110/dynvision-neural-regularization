"""
Epoch 0: CKA between the frozen pretrained CordsNet and monkey neural data.

For every batch of images (all from one recording day), the model runs 100 idle +
100 stimulus steps. Each layer's recorded activity (DynVision's `record`, i.e. after
the ReLU) is averaged over the 5 model steps of every 10 ms bin, flattened, and
compared with the z-scored MUA of the same images in the same bin. Activations are
used and discarded batch by batch; only CKA numbers are kept.

CKA: three estimators from Chun et al. 2025 (see cka_estimators.py): CKA_0 (naive =
usual linear CKA, the training loss form), CKA_S (Song, stimulus-corrected), CKA_C
(stimulus- and neuron-corrected).

Neural normalization: per day, per electrode, z-score with the mean / SD of that day's
train-split images, pooled over the 20 bins.

Outputs (in output_dir):
  results.npz          cka_batch      (n_batches, n_layers, n_areas, 20, 3)  CKA per batch
                                      (last axis: CKA_0, CKA_S, CKA_C)
                       cka_mean       (n_layers, n_areas, 20, 3)  mean over batches
                       cka_pooled     (n_layers, n_areas, 20, 3)  H values averaged over
                                      batches first, then the ratio (less biased, paper's CKA^N)
                       NaN = undefined (no variance across images, e.g. deep layers in the
                       first bins, still at their spontaneous state)
                       h_num          (n_batches, n_layers, n_areas, 20, 2)  numerator naive, Song
                       h_den_model    (n_batches, n_layers, 20, 3)  model denominators
                       h_den_neural   (n_batches, n_areas, 20, 3)  neural denominators
                       model_timecourse (n_layers, 20)  mean activity per layer and bin
                       batch_day      (n_batches,)
                       layers, areas, estimators, bin_start_ms
  batches.csv          batch, day, stim_idx of every image used
  summary.json         per area and estimator: mean CKA over bins for every layer, best layer
  resolved_config.yaml, git_commit.txt
results.npz is written last, so its existence means the run finished.

Usage:
  python run_epoch0.py --config configs/epoch0_pretrained.yaml
  python run_epoch0.py --config configs/epoch0_pretrained.yaml --max_batches 2 --device cpu \
      --output_dir /tmp/epoch0_smoke                                        # smoke test
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROUND1_DIR = Path(__file__).resolve().parents[1]      # .../CordsNet/Round1
CORDSNET_DIR = ROUND1_DIR.parent                       # .../CordsNet
REPO_ROOT = CORDSNET_DIR.parent                        # repository root
sys.path.insert(0, str(CORDSNET_DIR))
sys.path.insert(0, str(ROUND1_DIR))

from common.cordsnet_model import build_cordsnet, repeat_over_time  # noqa: E402
from common.manifest_dataset import ManifestDataset, build_transform  # noqa: E402
from cka_estimators import cka_all, h_cross, h_self, moments  # noqa: E402

N_BINS, STEPS_PER_BIN = 20, 5
ESTIMATORS = ["CKA_0", "CKA_S", "CKA_C"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"unknown ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_batches", type=int, default=None, help="only the first N batches (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg, model_cfg, eval_cfg, rt = cfg["data"], dict(cfg["model"]), cfg["eval"], cfg["runtime"]
    for key in ["output_dir", "device", "num_workers"]:
        if getattr(args, key) is not None:
            rt[key] = getattr(args, key)

    # config paths are relative to the repo root; a --output_dir given on the command line to the cwd
    out_dir = Path(args.output_dir).resolve() if args.output_dir else resolve(rt["output_dir"])
    if (out_dir / "results.npz").exists() and not args.overwrite:
        sys.exit(f"{out_dir / 'results.npz'} already exists; use --overwrite or another --output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(rt["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("runtime.device is cuda but no GPU is visible")
    layers, areas = eval_cfg["layers"], eval_cfg["areas"]

    # ---- split: day, row in that day's npz files, split of every image ----
    with open(resolve(data_cfg["split_csv"]), newline="") as fh:
        split_rows = list(csv.DictReader(fh))
    day_of = {int(r["stim_idx"]): int(r["day"]) for r in split_rows}
    row_of = {int(r["stim_idx"]): int(r["day_row"]) for r in split_rows}
    split_of = {int(r["stim_idx"]): r["split"] for r in split_rows}
    days = sorted(set(day_of.values()))

    # ---- neural data: z-score per day and electrode with train-split stats pooled over bins ----
    neural_dir, monkey = resolve(data_cfg["neural_dir"]), data_cfg["monkey"]
    neural = {}
    for day in days:
        for area in areas:
            d = np.load(neural_dir / f"{monkey}_day{day}_{area}_binned.npz")
            mua, idx = d["binned_mua"].astype(np.float32), d["train_idx"]
            assert mua.shape[0] == N_BINS, f"expected {N_BINS} bins, got {mua.shape[0]}"
            assert all(day_of[int(s)] == day and row_of[int(s)] == r for r, s in enumerate(idx)), \
                f"split CSV does not match {monkey} day {day} {area}"
            train_rows = np.array([split_of[int(s)] == "train" for s in idx])
            mean = mua[:, train_rows].mean(axis=(0, 1))
            sd = mua[:, train_rows].std(axis=(0, 1))
            assert (sd > 1e-6).all(), f"day {day} {area}: electrode with zero variance"
            neural[(day, area)] = (mua - mean) / sd
            print(f"neural day {day} {area}: {mua.shape[1]} trials, {mua.shape[2]} electrodes, "
                  f"stats from {train_rows.sum()} train images")

    # ---- batches: images of the eval splits, one day per batch, fixed shuffle, drop partial ----
    rng = np.random.default_rng(eval_cfg["seed"])
    B = eval_cfg["batch_size"]
    batches = []                                   # list of (day, [stim_idx, ...])
    for day in days:
        stims = sorted(s for s in day_of if day_of[s] == day and split_of[s] in eval_cfg["splits"])
        stims = rng.permutation(stims).tolist()
        for i in range(0, len(stims) - B + 1, B):
            batches.append((day, stims[i:i + B]))
    if args.max_batches:
        batches = batches[:args.max_batches]
    print(f"{len(batches)} batches of {B} images from splits {eval_cfg['splits']} "
          f"({sum(1 for b in batches for _ in b[1])} images)")

    # images: manifest rows of the TVSD train set, looked up by stim_idx
    dataset = ManifestDataset(resolve(data_cfg["manifest"]), splits=["train"],
                              transform=build_transform(data_cfg["image_size"]))
    pos = {s: k for k, s in enumerate(dataset.stim_idx)}
    loader = DataLoader(dataset, batch_sampler=[[pos[s] for s in b[1]] for b in batches],
                        num_workers=rt["num_workers"], pin_memory=(device.type == "cuda"))

    # ---- model ----
    checkpoint = resolve(model_cfg.pop("checkpoint"))
    n_classes = model_cfg.pop("n_classes")
    model, model_info = build_cordsnet(checkpoint, trainable=False, n_classes=n_classes,
                                       image_size=data_cfg["image_size"], **model_cfg)
    model = model.to(device)
    n_t = model_info["model_kwargs"]["n_timesteps"]
    assert n_t == N_BINS * STEPS_PER_BIN, "model stimulus steps must equal 20 bins x 5 steps"

    # Record every layer at every stimulus step through DynVision's own `record`
    # operation: wrap the single-step function, ask it for its responses outside the
    # idle period, and sum them into the current 10 ms bin.
    state = {}
    original_step = model._forward

    def step_with_record(x, t=None, feedforward_only=False, store_responses=False):
        if model._in_idle_period:
            return original_step(x, t=t, feedforward_only=feedforward_only, store_responses=False)
        out, resp = original_step(x, t=t, feedforward_only=feedforward_only, store_responses=True)
        k, sub = divmod(state["t"], STEPS_PER_BIN)
        for name in layers:
            r = resp[name]
            state["acc"][name] = r.clone() if sub == 0 else state["acc"][name] + r
            if sub == STEPS_PER_BIN - 1:
                a = (state["acc"][name] / STEPS_PER_BIN).flatten(1)
                state["mom"][(name, k)] = moments(a)
                state["mean"][(name, k)] = float(a.mean())
        state["t"] += 1
        return out, {}

    model._forward = step_with_record

    # ---- run ----
    nb, nl, na = len(batches), len(layers), len(areas)
    h_num = np.full((nb, nl, na, N_BINS, 2), np.nan)
    h_den_model = np.full((nb, nl, N_BINS, 3), np.nan)
    h_den_neural = np.full((nb, na, N_BINS, 3), np.nan)
    cka_batch = np.full((nb, nl, na, N_BINS, 3), np.nan)
    timecourse = np.zeros((nl, N_BINS))
    t0 = time.time()
    with torch.inference_mode():
        for b, (images, _) in enumerate(loader):
            day, stims = batches[b]
            state.update(t=0, acc={}, mom={}, mean={})
            model(repeat_over_time(images.to(device, non_blocking=True), n_t))
            assert state["t"] == n_t, f"recorded {state['t']} stimulus steps, expected {n_t}"

            rows = [row_of[s] for s in stims]
            for k in range(N_BINS):
                hy = {}
                for j, area in enumerate(areas):
                    my = moments(torch.from_numpy(neural[(day, area)][k, rows]).to(device))
                    hy[area] = (my, h_self(my))
                    h_den_neural[b, j, k] = [float(v) for v in hy[area][1]]
                for i, name in enumerate(layers):
                    mx = state["mom"][(name, k)]
                    hx = h_self(mx)
                    h_den_model[b, i, k] = [float(v) for v in hx]
                    timecourse[i, k] += state["mean"][(name, k)] / nb
                    for j, area in enumerate(areas):
                        my, hyy = hy[area]
                        h_num[b, i, j, k] = [float(v) for v in h_cross(mx, my)]
                        cka_batch[b, i, j, k] = [float(v) for v in cka_all(mx, my, hx, hyy)]
            if (b + 1) % rt["log_every"] == 0 or b + 1 == nb:
                el = time.time() - t0
                print(f"  batch {b + 1}/{nb} (day {day})  {el / 60:.1f} min, "
                      f"~{el / (b + 1) * (nb - b - 1) / 60:.1f} min left", flush=True)

    # ---- aggregate ----
    # NaN = undefined: no variance across images (deep layers in the first bins are
    # still at their spontaneous state), or a negative corrected denominator
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
    elapsed = time.time() - t0

    # ---- provenance (results.npz last) ----
    resolved = {"data": {**data_cfg, "n_batches": nb, "n_images": nb * B},
                "model": model_info, "eval": eval_cfg,
                "runtime": {**rt, "output_dir": str(out_dir), "max_batches": args.max_batches},
                "versions": {"torch": str(torch.__version__), "python": sys.version.split()[0]},
                "config_file": str(Path(args.config).resolve())}
    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    (out_dir / "git_commit.txt").write_text(git_commit() + "\n")
    with open(out_dir / "batches.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "day", "stim_idx"])
        for b, (day, stims) in enumerate(batches):
            w.writerows([b, day, s] for s in stims)

    # summary: CKA averaged over the 20 bins for every layer; CKA_0 from the batch mean
    # (same form as the training loss), CKA_S / CKA_C from the pooled H values (more stable)
    summary = {"n_batches": nb, "batch_size": B, "minutes": round(elapsed / 60, 2),
               "note": "CKA_0: mean over batches; CKA_S, CKA_C: pooled H over batches; "
                       "each value = mean over the 20 bins (undefined bins skipped)",
               "mean_over_bins_by_layer": {}, "best_layer": {}}
    for j, area in enumerate(areas):
        for e, est in enumerate(ESTIMATORS):
            table = cka_mean if est == "CKA_0" else cka_pooled
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_layer = np.nanmean(np.where(np.isfinite(table[:, j, :, e]), table[:, j, :, e], np.nan), axis=1)
            summary["mean_over_bins_by_layer"][f"{area} {est}"] = {
                name: round(float(v), 4) for name, v in zip(layers, per_layer)}
            summary["best_layer"][f"{area} {est}"] = (layers[int(np.nanargmax(per_layer))]
                                                       if np.isfinite(per_layer).any() else None)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    np.savez(out_dir / "results.npz", cka_batch=cka_batch, cka_mean=cka_mean, cka_pooled=cka_pooled,
             h_num=h_num, h_den_model=h_den_model, h_den_neural=h_den_neural,
             model_timecourse=timecourse, batch_day=np.array([d for d, _ in batches]),
             layers=np.array(layers), areas=np.array(areas), estimators=np.array(ESTIMATORS),
             bin_start_ms=np.arange(N_BINS) * 10)

    print(f"\nDone in {elapsed / 60:.1f} min. Saved to {out_dir}")
    for key, best in summary["best_layer"].items():
        vals = summary["mean_over_bins_by_layer"][key]
        print(f"  {key:10s} best {best}: " + "  ".join(f"{n[5:]}:{v:.3f}" for n, v in vals.items()))


if __name__ == "__main__":
    main()
