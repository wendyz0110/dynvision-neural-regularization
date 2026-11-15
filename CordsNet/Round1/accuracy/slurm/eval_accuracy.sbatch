#!/bin/bash
# Accuracy of all Round 1 checkpoints on the val split (forward pass only).
# 3 runs x 11 checkpoints x 1,854 images; ~20-30 min on one GPU.
#   sbatch --partition=h200_cds eval_accuracy.sbatch [extra eval_accuracy.py arguments]

#SBATCH --job-name=r1_accuracy
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
EXTRA="$*"

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)"
echo "extra: ${EXTRA}  code: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1/accuracy'
python -u eval_accuracy.py $EXTRA
"
echo "==== DONE $(date)"
