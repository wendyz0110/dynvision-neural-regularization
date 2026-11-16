#!/bin/bash
# Round 2 memory / speed test (fp32): the three-target config (V1-layer2, V4-layer6, IT-layer8)
# at batch 48 and 64, plus the Round 1 IT-only config at 64 for reference.
# Uses the shared Round1/train/memtest.py; results in CordsNet/Round2/outputs/.
#   sbatch --partition=h200_cds /scratch/wz2723/dynvision/DynVision/CordsNet/Round2/slurm/memtest_round2.sbatch

#SBATCH --job-name=r2_memtest
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
echo '--- three targets'
python -u memtest.py --config ../../Round2/configs/multi3_w1each.yaml --batch_sizes 48 64 --precisions fp32 \
    --out ../../Round2/outputs/memtest_multi3_job${SLURM_JOB_ID}.json
echo '--- Round 1 IT only (reference)'
python -u memtest.py --config configs/student_IT_layer8_w1.yaml --batch_sizes 64 --precisions fp32 \
    --out ../../Round2/outputs/memtest_ITonly_job${SLURM_JOB_ID}.json
"
echo "==== DONE $(date)"
