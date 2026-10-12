#!/bin/bash
# Submit one training job per config, then ONE evaluation job over all of
# them that starts only if every training job succeeds.
#
#   bash submit.sh configs/round2/control.yaml configs/round2/aligned_IT_w1.yaml ...
#
# Run from anywhere. Before a NEW set of runs: give each config a new
# runtime.output_dir, create the shared student init once
# (sbatch slurm/make_init.sbatch <one of the configs>), commit, push, and
# git pull here.
#
# For runs that are already trained, submit only the evaluation:
#   sbatch slurm/evaluate.sbatch <run_dir> <run_dir> ...

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
SLURM_DIR="$REPO/DiCarlo_experiment/slurm"
[ "$#" -ge 1 ] || { echo "usage: bash submit.sh <config.yaml> [<config.yaml> ...]"; exit 1; }

cd "$REPO/DiCarlo_experiment"
RUN_DIRS=()
for CONFIG in "$@"; do
    [ -f "$CONFIG" ] || { echo "ERROR: config $CONFIG not found."; exit 1; }
    RUN_DIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" \
              | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs)
    if [ -e "$RUN_DIR" ]; then
        echo "ERROR: $RUN_DIR already exists. Change output_dir in $CONFIG first."
        exit 1
    fi
    RUN_DIRS+=("$RUN_DIR")
done
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes (the jobs will run them):"
    git -C "$REPO" status --short --untracked-files=no
fi
mkdir -p /scratch/wz2723/dynvision/runs/slurm_logs

TRAIN_IDS=()
for CONFIG in "$@"; do
    ID=$(sbatch --parsable "$SLURM_DIR/train.sbatch" "$CONFIG")
    TRAIN_IDS+=("$ID")
    echo "train job $ID  <- $CONFIG"
done
DEP=$(IFS=:; echo "${TRAIN_IDS[*]}")
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$DEP" "$SLURM_DIR/evaluate.sbatch" "${RUN_DIRS[@]}")

echo "commit:   $(git -C "$REPO" rev-parse --short HEAD)"
echo "eval job $EVAL_ID (starts after ALL training jobs succeed)"
echo "status:   squeue -u $USER"
