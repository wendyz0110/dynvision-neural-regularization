#!/bin/bash
# Full training run (train_alignment.py). Usually submitted through submit.sh,
# which also queues the evaluation to start after this job succeeds.
#
# Direct use:  sbatch train.sbatch [config]      (default: configs/alignment_config.yaml)
#
#SBATCH --job-name=dicarlo_train
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LOG_DIR=/scratch/wz2723/dynvision/runs/slurm_logs
CONFIG=${1:-configs/alignment_config.yaml}     # relative to DiCarlo_experiment/

cd "$REPO/DiCarlo_experiment"

RUN_DIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" \
          | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs)

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR already exists. Change output_dir in $CONFIG" \
         "or remove the old folder, then resubmit."
    exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "config: $CONFIG | run dir: $RUN_DIR"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes:"
    git -C "$REPO" status --short --untracked-files=no
fi
nvidia-smi

GPU_LOG="$LOG_DIR/gpu_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 60 > "$GPU_LOG" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u train_alignment.py --config '$CONFIG'
"

# Keep a copy of the job's own log and the code version next to the results.
cp "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/logs/" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"

echo "==== DONE $(date)"
echo "GPU usage log: $GPU_LOG"
