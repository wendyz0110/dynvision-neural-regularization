#!/bin/bash
# Round 1 memory / speed test on one GPU. Optional first argument: a partition, e.g.
#   sbatch --partition=h200_public /scratch/.../CordsNet/Round1/train/slurm/memtest.sbatch
#   sbatch /scratch/.../CordsNet/Round1/train/slurm/memtest.sbatch          # any GPU

#SBATCH --job-name=r1_memtest
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)  code $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1/train'
python -u memtest.py --config configs/student_IT_layer8_w0p1.yaml \
    --batch_sizes 16 32 48 64 96 --precisions bf16 fp32 \
    --out ../outputs/round1/memtest_job${SLURM_JOB_ID}.json
"
echo "==== DONE $(date)"
