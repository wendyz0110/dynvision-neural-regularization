#!/bin/bash
# Round 1 training on one GPU. Resumes from <output_dir>/checkpoints/last.pt if present,
# so the same command can be chained (see submit_round1.sh).
#   sbatch train.sbatch configs/student_IT_layer8_w0p1.yaml [extra train.py arguments]

#SBATCH --job-name=r1_train
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CONFIG=$1
shift
EXTRA="$*"

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)"
echo "config: ${CONFIG}  extra: ${EXTRA}  code: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1/train'
python -u train.py --config '$CONFIG' $EXTRA
"
echo "==== DONE $(date)"
