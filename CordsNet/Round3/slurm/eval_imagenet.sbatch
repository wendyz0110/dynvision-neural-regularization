#!/bin/bash
# ImageNet-val accuracy (50,000 images) of the pretrained model and the Round 3 models.
# Forward pass only; ~20-30 min per model on one GPU (estimate).
#   sbatch --partition=h200_cds eval_imagenet.sbatch [extra eval_imagenet.py arguments]
# Quick check:
#   sbatch --partition=h200_cds --time=00:20:00 eval_imagenet.sbatch --models pretrained --max_batches 5 --out_dir CordsNet/Round3/outputs/imagenet_val_check

#SBATCH --job-name=r3_imagenet
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
IMAGENET_VAL=/projects/work/public/ml-datasets/imagenet/imagenet-val.sqf   # mounted at /imagenet/val
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
EXTRA="$*"

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)"
echo "extra: ${EXTRA}  code: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" --overlay "$IMAGENET_VAL:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round3/imagenet'
python -u eval_imagenet.py --num_workers ${SLURM_CPUS_PER_TASK} $EXTRA
"
echo "==== DONE $(date)"
