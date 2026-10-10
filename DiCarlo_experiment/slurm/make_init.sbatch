#!/bin/bash
# Create the shared student starting weights (make_student_init.py). Run ONCE
# before the first train job of a set of compared runs; every run's config
# points at the same student.init_checkpoint. Needs no GPU, but uses the same
# partition and environment as train.sbatch for simplicity (takes ~1 minute).
#
# Usage:  sbatch make_init.sbatch configs/round2/control.yaml
#         (any config of the set works: their student sections are identical)
#
#SBATCH --job-name=dicarlo_init
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CONFIG=${1:?usage: sbatch make_init.sbatch <config.yaml>}

echo "job $SLURM_JOB_ID on $(hostname) | start $(date) | config: $CONFIG"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u make_student_init.py --config '$CONFIG'
"
echo "==== DONE $(date)"
