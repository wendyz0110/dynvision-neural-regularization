"""
Create the shared starting weights for the student runs (run ONCE).

Every run that is compared (control, aligned_IT, aligned_V1V4IT, ...) loads
this same file, so they all start from bit-identical weights. Previously
train_alignment.py guaranteed this by building aligned and control together
in one process; with one student per run, the file takes over that role.

Reads only the `student` section of a run YAML (any of the compared runs'
configs works, since their student sections must be identical) and writes
student.init_checkpoint. The file stores the student spec, and
train_student.py refuses to start if its config's student section differs.

Usage (from DiCarlo_experiment/; no GPU needed):
    python make_student_init.py --config configs/student_control.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_student import TARGET_TRAINING_SEED, build_student, load_experiment_config, student_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", help="Building on CPU is fine.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    out = cfg.student.init_checkpoint
    if not out:
        raise ValueError("Set student.init_checkpoint in the config (where to write the file).")
    out = Path(out)
    if out.exists():
        raise FileExistsError(
            f"{out} already exists. Runs compared with each other must share it, so it is "
            f"never overwritten. Delete it by hand only if no run uses it."
        )
    seed = int(cfg.student.init_seed)
    if seed == TARGET_TRAINING_SEED:
        raise ValueError(f"student.init_seed must differ from the target's training seed ({TARGET_TRAINING_SEED}).")

    model = build_student(cfg, torch.device(args.device), seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "seed": seed,
        "student_spec": student_spec(cfg),
    }, out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved {out} (seed={seed}, {n_params:,} parameters).")


if __name__ == "__main__":
    main()