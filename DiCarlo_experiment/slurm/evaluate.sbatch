#!/bin/bash
# Joint evaluation of several finished train_student.py runs (evaluate_alignment.py):
# the target plus every run's student, on the test set.
# submit.sh queues it automatically after the training jobs. By hand:
#
#   sbatch evaluate.sbatch /scratch/wz2723/dynvision/runs/round2/control \
#                          /scratch/wz2723/dynvision/runs/round2/aligned_IT_w1  ...
#
# Results: <parent of the first run>/eval/<date-time>/ (never overwritten).
# Memory: features are held for one model and one layer at a time (V1 ~14 GB
# per timestep for 3925 images), so it fits on the 40 GB A100. Expected run
# time for 4 runs + target, 3 layers x 20 timesteps: ~1.5-2 h
# (V1 needs one pass per timestep: ~130 forward passes of the test set).
#
# Cluster: NYU burst (Google Cloud), account gm3239, partition c12m85-a100-1.
#
#SBATCH --job-name=dicarlo_eval
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

[ "$#" -ge 1 ] || { echo "usage: sbatch evaluate.sbatch <run_dir> [<run_dir> ...]"; exit 1; }
for RUN_DIR in "$@"; do
    if [ ! -f "$RUN_DIR/checkpoints/last.pt" ] || [ ! -f "$RUN_DIR/training_summary.json" ]; then
        echo "ERROR: $RUN_DIR has no last.pt / training_summary.json (training not finished?)."
        exit 1
    fi
done
RUNS=$(printf "'%s' " "$@")

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "runs: $*"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u evaluate_alignment.py --runs $RUNS
"

echo "==== DONE $(date)"
