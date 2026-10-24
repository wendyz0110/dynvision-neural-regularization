"""
Create the shared starting weights of the round 3 students, one file per seed.

All runs of one seed (control and every aligned condition) point at the same
student.init_checkpoint, so they start from bit-identical weights (paired
design). Different seeds point at different files.

Pass any number of round 3 student configs. Each distinct init_checkpoint
is created once from its student.init_seed. A file that already exists is
NOT overwritten: it is checked instead (same seed, same student spec) and
reused -- e.g. round 2's x4_seed2723.pt for seed 2723.

Usage (from DiCarlo_experiment/; no GPU needed):
    python round3/make_student_init.py --configs round3/configs/students/*.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROUND3_DIR = Path(__file__).resolve().parent
# round3/ must come BEFORE DiCarlo_experiment/: both contain a train_student.py
# (round 2's is one level up), and the round 3 one must win.
for _p in (ROUND3_DIR.parent, ROUND3_DIR):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import torch

from train_student import TARGET_TRAINING_SEED, build_student, load_experiment_config, student_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--device", default="cpu", help="Building on CPU is fine.")
    args = parser.parse_args()

    # init file -> (seed, spec, configs using it)
    wanted = {}
    for path in args.configs:
        cfg = load_experiment_config(path)
        out = cfg.student.init_checkpoint
        if not out:
            raise ValueError(f"{path}: set student.init_checkpoint.")
        seed, spec = int(cfg.student.init_seed), student_spec(cfg)
        if seed == TARGET_TRAINING_SEED:
            raise ValueError(f"{path}: student.init_seed must differ from the target's seed ({TARGET_TRAINING_SEED}).")
        if out in wanted:
            if (wanted[out]["seed"], wanted[out]["spec"]) != (seed, spec):
                raise ValueError(f"{path} uses {out} with a different seed or student spec than "
                                 f"{wanted[out]['configs'][0]}.")
            wanted[out]["configs"].append(path)
        else:
            wanted[out] = {"seed": seed, "spec": spec, "cfg": cfg, "configs": [path]}

    for out, w in wanted.items():
        out_path = Path(out)
        n = len(w["configs"])
        if out_path.exists():
            existing = torch.load(out_path, map_location="cpu", weights_only=False)
            if int(existing.get("seed", -1)) != w["seed"] or existing.get("student_spec") != w["spec"]:
                raise RuntimeError(f"{out_path} exists but was made with seed {existing.get('seed')} / a different "
                                   f"student spec. It is never overwritten; use another init_checkpoint path.")
            print(f"[reuse] {out_path} (seed {w['seed']}) -- matches, used by {n} config(s)")
            continue
        model = build_student(w["cfg"], torch.device(args.device), seed=w["seed"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "seed": w["seed"],
            "student_spec": w["spec"],
        }, out_path)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[new]   {out_path} (seed {w['seed']}, {n_params:,} parameters) -- used by {n} config(s)")


if __name__ == "__main__":
    main()
