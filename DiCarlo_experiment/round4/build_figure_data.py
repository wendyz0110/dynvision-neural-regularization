"""
Build the round 4 figure tables (same 5 tables as round 3) and a shareable package.

Reads the raw round 4 outputs on burst and writes

    <out>/figure_data/            round 4 tables + target_model_card.yaml
    <out>/figure_data_round3/     copy of round 3's figure_data (for the round 3 vs 4 comparison)
    <out>.tar.gz                  the whole folder, to download

The tables have exactly the columns of round 3 (see the "Data tables" cell of
plot_round3_results.ipynb), so both rounds are read with the same notebook code.

Usage (from DiCarlo_experiment/, inside the dynvision environment):
    python round4/build_figure_data.py
    python round4/build_figure_data.py --target target_x8_table1 --out /scratch/.../share_x8
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/scratch/wz2723/dynvision/runs")
T = 20
LAST_T = 19


def split_name(name: str):
    """'IT_w0p1_s2724' -> ('IT_w0p1', 2724)."""
    condition, seed = name.rsplit("_s", 1)
    return condition, int(seed)


def target_table(target_dir: Path) -> pd.DataFrame:
    m = pd.read_csv(target_dir / "metrics" / "metrics.csv")
    # Row epochs_completed 0 = initial weights: no training epoch yet, so its
    # training loss is the eval-mode loss on the training set.
    train_ce = m["train_ce_loss"].where(m["epochs_completed"] > 0, m["train_eval_ce_loss"])
    tt = pd.DataFrame({"epochs_completed": m["epochs_completed"], "train_ce": train_ce,
                       "val_ce": m["val_ce_loss"], "train_accuracy_eval": m["train_eval_accuracy"],
                       "val_accuracy": m["val_accuracy"]})
    for t in range(T):
        tt[f"val_acc_t{t:02d}"] = m[f"val_acc_t{t:02d}"]
    return tt


def student_table(students_dir: Path) -> pd.DataFrame:
    rename = {"val_ce_loss": "val_ce", "val_total_loss": "val_total_loss",
              "val_target_agreement": "val_same_answer_as_target", "val_gt_accuracy": "val_accuracy",
              "diag_grad_ce": "grad_ce", "diag_grad_cka_sum": "grad_cka_sum",
              "diag_ratio_cka_to_ce": "grad_ratio_cka_to_ce", "train_grad_norm": "train_grad_norm",
              "train_clip_fraction": "train_clip_fraction", "learning_rate": "learning_rate"}
    parts = []
    for run in sorted(students_dir.iterdir()):
        f = run / "metrics" / "metrics.csv"
        if not f.exists():
            print(f"WARNING: {run.name} has no metrics.csv, skipped")
            continue
        condition, seed = split_name(run.name)
        m = pd.read_csv(f)
        if not (m["epochs_completed"] == 0).any():
            print(f"WARNING: {run.name} has no epochs_completed 0 row")
        cols = {k: v for k, v in rename.items() if k in m.columns}
        cols.update({c: c for c in m.columns if c.startswith("val_cka_")})
        cols.update({c: c.replace("diag_grad_cka_", "grad_cka_") for c in m.columns
                     if c.startswith("diag_grad_cka_") and c != "diag_grad_cka_sum"})
        p = m[list(cols)].rename(columns=cols)
        p.insert(0, "epochs_completed", m["epochs_completed"])
        p.insert(0, "seed", seed)
        p.insert(0, "condition", condition)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def test_tables(eval_dir: Path):
    behavior, cka_parts, have_target = [], [], False
    for e in sorted(eval_dir.glob("seed*")):
        m = pd.read_csv(e / "metrics_over_time.csv")
        for model, d in m.groupby("model"):
            if model == "target":
                if have_target:                     # identical in every evaluation: keep it once
                    continue
                condition, seed, have_target = "target", np.nan, True
            else:
                condition, seed = split_name(model)
            behavior.append(pd.DataFrame({"model": model, "condition": condition, "seed": seed,
                                          "timestep": d["timestep"], "accuracy": d["gt_accuracy"],
                                          "same_answer_as_target": d["agreement"],
                                          "ce_vs_target_answer": d["ce_pseudo"]}))
        c = pd.read_csv(e / "cka_over_time.csv")
        c = c[c["model_b"] == "target"]
        for model, d in c.groupby("model_a"):
            condition, seed = split_name(model)
            cka_parts.append(pd.DataFrame({"comparison": "student_vs_target", "condition": condition, "seed": seed,
                                           "other_seed": np.nan, "layer": d["layer"], "timestep": d["timestep"],
                                           "cka": d["cka_biased"]}))
    cross = eval_dir / "controls_across_seeds" / "cka_over_time.csv"
    if cross.exists():
        c = pd.read_csv(cross)
        c = c[(c["model_a"] != "target") & (c["model_b"] != "target")]
        for (a, b), d in c.groupby(["model_a", "model_b"]):
            cka_parts.append(pd.DataFrame({"comparison": "control_vs_control", "condition": "control",
                                           "seed": split_name(a)[1], "other_seed": split_name(b)[1],
                                           "layer": d["layer"], "timestep": d["timestep"], "cka": d["cka_biased"]}))
    else:
        print("WARNING: no controls_across_seeds evaluation -> no seed baseline in the CKA figures")
    return pd.concat(behavior, ignore_index=True), pd.concat(cka_parts, ignore_index=True)


def summary_table(test_behavior: pd.DataFrame, test_cka: pd.DataFrame) -> pd.DataFrame:
    b = test_behavior[test_behavior["timestep"] == LAST_T]
    c = test_cka[(test_cka["comparison"] == "student_vs_target") & (test_cka["timestep"] == LAST_T)]
    c = c.pivot_table(index=["condition", "seed"], columns="layer", values="cka").add_prefix("cka_").reset_index()
    return b.drop(columns=["model", "timestep"]).merge(c, on=["condition", "seed"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="target_x8_table1")
    ap.add_argument("--round3_data", default=str(RUNS / "round3" / "figure_data"))
    ap.add_argument("--out", default=None, help="default: runs/round4/share_<target>")
    args = ap.parse_args()

    r4 = RUNS / "round4"
    target_dir = r4 / args.target
    students_dir = r4 / f"students_{args.target}"
    eval_dir = r4 / f"eval_{args.target}"
    out = Path(args.out) if args.out else r4 / f"share_{args.target}"
    data = out / "figure_data"
    if out.exists():
        raise SystemExit(f"{out} already exists: remove it first (rm -r {out}) or pass --out")
    data.mkdir(parents=True)

    target_table(target_dir).to_csv(data / "target_training.csv", index=False)
    shutil.copy(target_dir / "model_card.yaml", data / "target_model_card.yaml")
    st = student_table(students_dir)
    st.to_csv(data / "student_training.csv", index=False)
    test_behavior, test_cka = test_tables(eval_dir)
    test_behavior.to_csv(data / "test_behavior.csv", index=False)
    test_cka.to_csv(data / "test_cka.csv", index=False)
    summary = summary_table(test_behavior, test_cka)
    summary.to_csv(data / "summary.csv", index=False)

    r3 = Path(args.round3_data)
    if r3.is_dir():
        shutil.copytree(r3, out / "figure_data_round3")
    else:
        print(f"WARNING: round 3 tables not found at {r3} -> no round 3 comparison in the package")

    # ---- checks ----
    print(f"\nTables written to {data}")
    for f in sorted(data.glob("*.csv")):
        print(f"  {f.name:24s} {len(pd.read_csv(f)):6d} rows")
    seeds = st.groupby("condition")["seed"].nunique().to_dict()
    epochs = st.groupby(["condition", "seed"])["epochs_completed"].max()
    print("seeds per condition:", seeds)
    print("epochs completed: min", epochs.min(), "max", epochs.max())
    print("evaluated students:", summary[summary["condition"] != "target"].groupby("condition")["seed"].nunique().to_dict())
    print("control_vs_control pairs:", test_cka.loc[test_cka["comparison"] == "control_vs_control",
                                                    ["seed", "other_seed"]].drop_duplicates().shape[0])

    tar_path = out.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print(f"\nPackage: {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()