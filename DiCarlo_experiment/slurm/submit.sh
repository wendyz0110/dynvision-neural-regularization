#!/bin/bash
# Submit training, then evaluation that starts only if training succeeds.
#
#   bash submit.sh [config]      (default: configs/alignment_config.yaml)
#
# Run from anywhere. Before submitting a NEW run, change runtime.output_dir
# in the config (alignment_run_01 -> 02 ...), commit, push, and git pull here.

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
SLURM_DIR="$REPO/DiCarlo_experiment/slurm"
CONFIG=${1:-configs/alignment_config.yaml}

cd "$REPO/DiCarlo_experiment"
[ -f "$CONFIG" ] || { echo "ERROR: config $CONFIG not found."; exit 1; }

RUN_DIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" \
          | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs)

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR already exists. Change output_dir in $CONFIG first."
    exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes (the job will run them):"
    git -C "$REPO" status --short --untracked-files=no
fi

mkdir -p /scratch/wz2723/dynvision/runs/slurm_logs

TRAIN_ID=$(sbatch --parsable "$SLURM_DIR/train.sbatch" "$CONFIG")
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" "$SLURM_DIR/evaluate.sbatch" "$RUN_DIR")

echo "config:   $CONFIG"
echo "run dir:  $RUN_DIR"
echo "commit:   $(git -C "$REPO" rev-parse --short HEAD)"
echo "train job $TRAIN_ID  -> evaluation job $EVAL_ID (starts after training succeeds)"
echo "status:   squeue -u $USER"
