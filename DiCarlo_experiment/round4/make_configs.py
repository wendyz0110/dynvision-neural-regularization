"""
Generate the round 4 student configs: every CONDITION x every SEED.

Round 4 = round 3's design (same 6 conditions, same 3 seeds, same init files,
same recipe) with a different TARGET: DyRCNNx8 (target_x8) instead of the
DyRCNNx4 target_r3. The students stay DyRCNNx4. Because the init files and
data order are the same as in round 3, every round 4 run is also paired with
the round 3 run of the same condition and seed: only the target differs.

Edit CONDITIONS / SEEDS / SHARED below, run this script, commit the YAMLs.
Each YAML is complete on its own (nothing is inherited at run time); this
script only saves writing 18 near-identical files by hand.

Paired design: within one seed, all conditions share the init file, the
data order and the runtime seed -- they differ only in alignment.layers.

Usage (from DiCarlo_experiment/):
    python round4/make_configs.py            # writes round4/configs/students/*.yaml
    python round4/make_configs.py --check    # only list what would be written
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

OUT_DIR = Path(__file__).resolve().parent / "configs" / "students"

SEEDS = [2723, 2724, 2725]          # never 7000 (the target's seed)

# name -> (description, loss layers). Each layer: (layer, timestep, weight);
# timestep -1 = last (t19).
CONDITIONS = {
    "control": ("CE only (no alignment term)", []),
    "IT_w1": ("CE + 1.0 * log(1 - CKA_IT@t19)", [("IT", -1, 1.0)]),
    "IT_w0p1": ("CE + 0.1 * log(1 - CKA_IT@t19)", [("IT", -1, 0.1)]),
    "V1V4IT_w0p1": ("CE + 0.1 * sum over V1, V4, IT of log(1 - CKA@t19)  (total weight 0.3)",
                    [("V1", -1, 0.1), ("V4", -1, 0.1), ("IT", -1, 0.1)]),
    "V1V4IT_total0p1": ("CE + (0.1/3) * sum over V1, V4, IT of log(1 - CKA@t19)  (total weight 0.1)",
                        [("V1", -1, 0.1 / 3), ("V4", -1, 0.1 / 3), ("IT", -1, 0.1 / 3)]),
    "IT_multiT_total0p1": ("CE + 0.025 * sum over t in {2, 6, 12, 19} of log(1 - CKA_IT@t)  (total weight 0.1)",
                           [("IT", 2, 0.025), ("IT", 6, 0.025), ("IT", 12, 0.025), ("IT", -1, 0.025)]),
}

# CKA logged (no gradient) in EVERY run, so all conditions report the same
# numbers. Must exist in the target cache.
MONITOR = [("V1", -1), ("V4", -1), ("IT", -1), ("IT", 2), ("IT", 6), ("IT", 12)]

PATHS = {
    "image_dir": "/scratch/wz2723/dynvision/cache/imagenette_images",
    "target_dir": "/scratch/wz2723/dynvision/cache/targets/target_x8",
    "data_root": "/scratch/wz2723/dynvision/data/imagenette2-320",
    "split_file": "/scratch/wz2723/dynvision/DynVision/DiCarlo_experiment/runs/alignment_run_01/data/split.json",
    "init_template": "/scratch/wz2723/dynvision/cache/student_init/x4_seed{seed}.pt",   # = round 3's files
    "runs": "/scratch/wz2723/dynvision/runs/round4/students",
}

# Shared recipe. [R3-SHARED] = changed from round 2 together with the target.
SHARED_OPTIMIZATION = {
    "epochs": 200,
    "optimizer": "adamw",              # [R3-SHARED] round 2: adam
    "learning_rate": 1.41e-4,          # 2e-4 * sqrt(32 / 64)
    "weight_decay": 0.01,              # [R3-SHARED] decoupled; round 2: 5e-4 coupled
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "lr_group_factors": {"recurrence": 0.2, "feedback": 0.5},
    "accumulate_grad_batches": 1,
    "max_grad_norm": 500.0,            # [R3-SHARED] round 2: 1.0
    "use_cosine_scheduler": True,
    "cosine_t_max": 200,               # [R3-SHARED] round 2: 500
}


def layer_entry(layer: str, timestep: int, weight: float | None = None) -> dict:
    entry = {"name": layer, "timestep": timestep}
    if weight is not None:
        entry["weight"] = float(weight)
    return entry


def build_config(condition: str, seed: int) -> dict:
    description, layers = CONDITIONS[condition]
    name = f"{condition}_s{seed}"
    return {
        "experiment": {
            "role": "control" if not layers else "aligned",
            "description": f"{description} | seed {seed}",
        },
        "cache": {"image_dir": PATHS["image_dir"], "target_dir": PATHS["target_dir"]},
        "data": {
            "root": PATHS["data_root"],
            "split_file": PATHS["split_file"],
            "batch_size": 32,
            "input_timesteps": 20,
            "shuffle_seed": seed,
            "num_workers": 8,
        },
        "student": {
            # Architecture = utils/student_model_utils.DEFAULT_STUDENT_KWARGS
            # (= round 3 students; the round 4 TARGET is DyRCNNx8).
            "init_checkpoint": PATHS["init_template"].format(seed=seed),
            "init_seed": seed,
        },
        "alignment": {
            "layers": [layer_entry(*l) for l in layers],
            "monitor_layers": [layer_entry(*m) for m in MONITOR],
            "ce_weight": 1.0,
            "ce_first_timestep": 4,
            "pseudo_label_timestep": -1,
            "warmup_epochs": 0,
            "min_cka_batch_size": 16,
        },
        "optimization": dict(SHARED_OPTIMIZATION),
        "runtime": {
            "seed": seed,
            "device": "cuda",
            "precision": "bfloat16",
            "output_dir": f"{PATHS['runs']}/{name}",
            "log_every_n_steps": 20,
        },
        "checkpoint": {"monitor": "val_total_loss", "mode": "min", "save_every_n_epochs": 10, "eval_init": True},
        "diagnostics": {"gradient_terms": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="List the configs, write nothing.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            name = f"{condition}_s{seed}"
            path = OUT_DIR / f"{name}.yaml"
            cfg = build_config(condition, seed)
            header = (f"# Round 4 student: {name}  (target: DyRCNNx8 target_x8)\n"
                      f"# {CONDITIONS[condition][0]}\n"
                      f"# GENERATED by round4/make_configs.py -- edit that script, not this file.\n\n")
            if not args.check:
                path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=120), encoding="utf-8")
            written.append(path)
    print(f"{'Would write' if args.check else 'Wrote'} {len(written)} configs to {OUT_DIR}:")
    for p in written:
        print("  ", p.name)


if __name__ == "__main__":
    main()
