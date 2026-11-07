#!/bin/bash
# Epoch 0: CKA of the frozen pretrained CordsNet-R8 vs monkey N neural data.
#
# Before the first submission, create the log folder (Slurm cannot create it):
#   mkdir -p /scratch/wz2723/cordsnet_runs/slurm_logs
# Submit from anywhere:
#   sbatch /scratch/wz2723/dynvision/DynVision/CordsNet/Round1/epoch0/slurm/run_epoch0.sbatch

#SBATCH --job-name=r1_epoch0
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CONFIG=${1:-configs/epoch0_pretrained.yaml}

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)"
echo "config: ${CONFIG}   code: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1/epoch0'
python -u run_epoch0.py --config '$CONFIG' --num_workers ${SLURM_CPUS_PER_TASK}
"

echo "==== DONE $(date)"
