#!/bin/bash
# Round 4: train the THINGS head over all 1854 concepts on the stored features (modes all / frozen x weight decay grid).
#   Run after: cd CordsNet/Round3/slurm && sbatch --partition=h200_public things_features.sbatch --concepts all --out_dir CordsNet/Round4/outputs/things_head
#   cd CordsNet/Round4/slurm && sbatch --partition=h200_public things_head.sbatch [extra train_head.py arguments]

#SBATCH --job-name=r4_things_head
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
cd '$REPO/CordsNet/Round4/things_head'
python -u train_head.py $EXTRA
"
echo "==== DONE $(date)"
