#!/bin/bash
# Create the round 4 student init files, one per seed (round4/make_student_init.py).
# Existing files (e.g. round 2's x4_seed2723.pt) are checked and reused, never
# overwritten. Run ONCE before submit_students.sh. Needs no GPU but uses the
# same partition and environment as training for simplicity (~1-2 minutes).
#
# Usage (from the repo root, /scratch/wz2723/dynvision/DynVision):
#   sbatch DiCarlo_experiment/round4/slurm/make_inits.sbatch
#   (default: every config in round4/configs/students_target_x8_table1/; or pass configs,
#    relative to DiCarlo_experiment/)
#
#SBATCH --job-name=r4_inits
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

if [ "$#" -ge 1 ]; then CONFIGS=$(printf "'%s' " "$@"); else CONFIGS="round4/configs/students_target_x8_table1/*.yaml"; fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u round4/make_student_init.py --configs $CONFIGS
"
ls -la /scratch/wz2723/dynvision/cache/student_init/
echo "==== DONE $(date)"
