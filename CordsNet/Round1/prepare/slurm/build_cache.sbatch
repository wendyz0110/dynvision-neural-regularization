#!/bin/bash
# One-time Round 1 cache (images, z-scored neural data, pseudo-labels). CPU only.
#   sbatch /scratch/wz2723/dynvision/DynVision/CordsNet/Round1/prepare/slurm/build_cache.sbatch

#SBATCH --job-name=r1_cache
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)  code $(git -C "$REPO" rev-parse --short HEAD)"
singularity exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1/prepare'
python -u build_cache.py \
    --split_csv ../splits/monkeyN_split.csv \
    --manifest /scratch/wz2723/data/THINGS/things_manifest_tvsd.csv \
    --neural_dir /scratch/wz2723/data/TVSD/monkeyN_clean \
    --pseudo_labels /scratch/wz2723/cordsnet_runs/inference/things_cordsnetr8/outputs.npz \
    --out_dir /scratch/wz2723/data/TVSD/round1_cache_monkeyN \
    --num_workers ${SLURM_CPUS_PER_TASK}
"
echo "==== DONE $(date)"
