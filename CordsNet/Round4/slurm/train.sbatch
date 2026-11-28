#!/bin/bash
# Round 4 training on one GPU (ImageNet train mounted at /imagenet/train). Resumes from
# <output_dir>/checkpoints/last.pt if present, so the same command can be chained (see submit_round4.sh).
#   cd CordsNet/Round4/slurm && sbatch --partition=h200_cds,h200_public train.sbatch ../configs/control.yaml [extra train.py arguments]
# Batch 96 needs an H200 (~142 GB). (The config path is relative to CordsNet/Round4/train, where train.py runs)

#SBATCH --job-name=r4_train
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
IMAGENET_TRAIN=/projects/work/public/ml-datasets/imagenet/imagenet-train.sqf   # mounted at /imagenet/train
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CONFIG=$1
shift
EXTRA="$*"

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)"
echo "config: ${CONFIG}  extra: ${EXTRA}  code: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" --overlay "$IMAGENET_TRAIN:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round4/train'
python -u train.py --config '$CONFIG' --num_workers 8 $EXTRA
"
echo "==== DONE $(date)"
