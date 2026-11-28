#!/bin/bash
# Round 4 memory / speed test (one full training step: ImageNet part + THINGS part) for B = 96.
#   cd CordsNet/Round4/slurm && sbatch --partition=h200_cds,h200_public memtest.sbatch [extra memtest.py arguments]

#SBATCH --job-name=r4_memtest
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
EXTRA="$*"

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)  code $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round4/train'
python -u memtest.py --config ../configs/multi3_w1each.yaml --batch_sizes 96 \
    --out ../outputs/memtest/memtest_${SLURM_JOB_ID}.json $EXTRA
"
echo "==== DONE $(date)"
