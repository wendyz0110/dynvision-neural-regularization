#!/bin/bash
# "Epoch 0" (before training) measurements for the training curves
# (round3/eval_init.py): every student run in round3/configs/students/ and
# the target. Writes <run dir>/metrics/init_metrics.csv; touches nothing else.
# Needs the runs to be trained. ~10-15 minutes.
#
# Usage (from the repo root, /scratch/wz2723/dynvision/DynVision):
#   sbatch DiCarlo_experiment/round3/slurm/eval_init.sbatch
#
#SBATCH --job-name=r3_evalinit
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u round3/eval_init.py \
    --target_config round3/configs/target_r3.yaml \
    --configs round3/configs/students/*.yaml
"
echo "==== DONE $(date)"
