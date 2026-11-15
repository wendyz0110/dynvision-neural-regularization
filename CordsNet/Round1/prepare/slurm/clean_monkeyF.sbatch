#!/bin/bash
# Monkey F: copy the binned files with V4 and IT swapped back (see clean_monkeyF.py). CPU only, a few minutes.
#   sbatch /scratch/wz2723/dynvision/DynVision/CordsNet/Round1/prepare/slurm/clean_monkeyF.sbatch

#SBATCH --job-name=clean_monkeyF
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
cd '$REPO/CordsNet/Round1'
python -u clean_monkeyF.py \
    --raw_dir /scratch/wz2723/data/TVSD/monkeyF \
    --out_dir /scratch/wz2723/data/TVSD/monkeyF_clean
"
echo "==== DONE $(date)"
