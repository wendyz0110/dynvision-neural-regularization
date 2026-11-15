#!/bin/bash
# Round 1 checkpoints (trained on monkey N) evaluated on monkey F: 3 runs x epochs 0-10, val split.
# Finished epochs are skipped, so resubmit if the job runs out of time.
#   sbatch --partition=h200_cds eval_other_monkey.sbatch [extra eval_other_monkey.py arguments]

#SBATCH --job-name=r1_eval_monkeyF
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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
cd '$REPO/CordsNet/Round1/transfer'
python -u eval_other_monkey.py $EXTRA
"
echo "==== DONE $(date)"
