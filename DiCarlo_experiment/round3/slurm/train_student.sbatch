#!/bin/bash
# Train ONE round 3 student (round3/train_student.py). submit_students.sh submits
# all of them; by hand:
#
# Cluster: NYU burst, account gm3239, partition c12m85-a100-1
# (1x A100 40 GB, 12 CPUs, 85 GB RAM per node).
#
# Usage (from DiCarlo_experiment/ on burst):
#   sbatch round3/slurm/train_student.sbatch round3/configs/students/control_s2723.yaml           # new run
#   sbatch round3/slurm/train_student.sbatch round3/configs/students/control_s2723.yaml resume    # continue from
#                                                                                   # <run dir>/checkpoints/last.pt
# Config paths are relative to DiCarlo_experiment/.
#
#SBATCH --job-name=r3_student
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LOG_DIR=/scratch/wz2723/dynvision/runs/slurm_logs
CONFIG=${1:?usage: sbatch round3/slurm/train_student.sbatch <config.yaml> [resume]}
MODE=${2:-new}

cd "$REPO/DiCarlo_experiment"

RUN_DIR=$(grep -E '^[[:space:]]*output_dir:' "$CONFIG" \
          | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs)

case "$MODE" in
    new)
        if [ -e "$RUN_DIR" ]; then
            echo "ERROR: $RUN_DIR already exists. Change output_dir in $CONFIG," \
                 "remove the old folder, or resubmit with 'resume'."
            exit 1
        fi
        RESUME_FLAG="" ;;
    resume)
        RESUME_FLAG="--resume" ;;
    *)
        echo "ERROR: second argument must be 'resume' (or nothing), got '$MODE'"; exit 1 ;;
esac

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "config: $CONFIG | run dir: $RUN_DIR | mode: $MODE"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes:"
    git -C "$REPO" status --short --untracked-files=no
fi
nvidia-smi

mkdir -p "$LOG_DIR"
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
python -u round3/train_student.py --config '$CONFIG' $RESUME_FLAG
"

# Keep a copy of the job's own log, the config and the code version next to the results.
mkdir -p "$RUN_DIR/logs"
cp "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/logs/" 2>/dev/null || true
cp "$CONFIG" "$RUN_DIR/" 2>/dev/null || true
git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"

echo "==== DONE $(date)"
echo "GPU usage log: $GPU_LOG"
