#!/bin/bash
# Smoke test: 1 training epoch (configs/smoke_config.yaml), then evaluation of
# that run. Records GPU memory/utilization every 10 s.
#
# Before the FIRST submission, create the log folder once:
#   mkdir -p /scratch/wz2723/dynvision/runs/slurm_logs
# Submit from anywhere:
#   sbatch /scratch/wz2723/dynvision/DynVision/DiCarlo_experiment/slurm/smoke_test.sbatch
#
#SBATCH --job-name=dicarlo_smoke
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

# ---- paths -----------------------------------------------------------------
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LOG_DIR=/scratch/wz2723/dynvision/runs/slurm_logs
CONFIG=configs/smoke_config.yaml          # relative to DiCarlo_experiment/
EXPECTED_TARGET_ACC=0.67                  # target's reported test accuracy

cd "$REPO/DiCarlo_experiment"

# Run folder = runtime.output_dir from the config (so the two cannot disagree).
RUN_DIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" \
          | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs)

# Refuse to reuse a run folder: it would overwrite checkpoints and mix CSVs.
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR already exists. Change output_dir in $CONFIG" \
         "or remove the old folder, then resubmit."
    exit 1
fi

# ---- record what is being run ------------------------------------------------
echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "config: $CONFIG | run dir: $RUN_DIR"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes:"
    git -C "$REPO" status --short --untracked-files=no
fi
nvidia-smi

# ---- GPU monitor in the background (stopped automatically at exit) ------------
GPU_LOG="$LOG_DIR/gpu_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 10 > "$GPU_LOG" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

# ---- train, then evaluate -------------------------------------------------------
singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'

echo '==== TRAINING  ' \$(date)
python -u train_alignment.py --config '$CONFIG'

echo '==== EVALUATION ' \$(date)
python -u evaluate_alignment.py \
    --run_dir '$RUN_DIR' \
    --name smoke \
    --expected_target_accuracy $EXPECTED_TARGET_ACC
"

echo "==== DONE $(date)"
echo "GPU usage log: $GPU_LOG"
