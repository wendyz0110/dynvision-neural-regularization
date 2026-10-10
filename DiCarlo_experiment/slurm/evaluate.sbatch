#!/bin/bash
# Evaluation of a finished training run (evaluate_alignment.py).
# submit.sh queues it automatically after training. To (re-)run it by hand:
#
#   sbatch evaluate.sbatch /scratch/wz2723/dynvision/runs/alignment_run_01
#
# Every run writes a new folder <run_dir>/eval/<date-time>/; earlier
# evaluations are never overwritten.
#
# Cluster: NYU burst (Google Cloud), account gm3239, partition c12m85-a100-1
# (1x A100 40 GB, 12 CPUs, 85 GB RAM per node; the whole node is allocated).
# The Torch version of this script is in git history (commit d26c953).
# NOTE: on Torch (H200) evaluation peaked at ~91 GB GPU memory, so it will
# not fit on a 40 GB A100 until the evaluation batch size can be lowered.
#
#SBATCH --job-name=dicarlo_eval
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
EXPECTED_TARGET_ACC=0.67

RUN_DIR=${1:?usage: sbatch evaluate.sbatch <run_dir>}

if [ ! -f "$RUN_DIR/checkpoints/aligned_last.pt" ] || [ ! -f "$RUN_DIR/checkpoints/control_last.pt" ]; then
    echo "ERROR: $RUN_DIR has no aligned_last.pt / control_last.pt (training not finished?)."
    exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "run dir: $RUN_DIR"
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
python -u evaluate_alignment.py \
    --run_dir '$RUN_DIR' \
    --expected_target_accuracy $EXPECTED_TARGET_ACC
"

echo "==== DONE $(date)"
