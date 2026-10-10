#!/bin/bash
# Build the image cache and the target cache (prepare_images.py ->
# prepare_targets.py). Same environment setup as train.sbatch.
#
# Cluster: NYU burst (Google Cloud), account gm3239, partition c12m85-a100-1.
#
# Usage:
#   sbatch prepare_cache.sbatch dry_run   (default) image cache + target DRY RUN:
#                                         prints real shapes / sizes / time,
#                                         writes no target files
#   sbatch prepare_cache.sbatch full      image cache (skipped if it exists)
#                                         + full target cache
#
# The image cache is built in BOTH modes: prepare_targets reads from it, and
# it is small (~2 GB) and needed anyway. If it already exists it is reused.
#
#SBATCH --job-name=dicarlo_prepcache
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

MODE=${1:-dry_run}

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

DATA_ROOT=/scratch/wz2723/dynvision/data/imagenette2-320
IMAGE_CACHE=/scratch/wz2723/dynvision/cache/imagenette_images
TARGET_CACHE=/scratch/wz2723/dynvision/cache/targets/x4_surrogate

# Target checkpoint. Empty = load_target_model's default (repo trained.pt).
# Setting the full path is recommended: its hash goes into the manifest.
CHECKPOINT="/scratch/wz2723/dynvision/DynVision/DiCarlo_experiment/x4_generators_tfb12/x4_pointdepthwise_tfb12/trained.pt"

LAYERS="V1 V4 IT"
TIMESTEPS="-1"

case "$MODE" in
    dry_run) TARGET_FLAGS="--dry_run" ;;
    full)    TARGET_FLAGS="" ;;
    *) echo "ERROR: mode must be 'dry_run' or 'full', got '$MODE'"; exit 1 ;;
esac
CKPT_FLAG=""
if [ -n "$CHECKPOINT" ]; then
    CKPT_FLAG="--checkpoint '$CHECKPOINT'"
fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date) | mode: $MODE"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes:"
    git -C "$REPO" status --short --untracked-files=no
fi
nvidia-smi

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'

echo '==== [1/2] image cache'
if [ -f '$IMAGE_CACHE/manifest.json' ]; then
    echo 'Image cache already exists, reusing: $IMAGE_CACHE'
else
    python -u prepare_images.py \
        --data_root '$DATA_ROOT' \
        --out_dir '$IMAGE_CACHE' \
        --num_workers 10
fi

echo '==== [2/2] target cache ($MODE)'
python -u prepare_targets.py \
    --image_cache '$IMAGE_CACHE' \
    --out_dir '$TARGET_CACHE' \
    --layers $LAYERS \
    --timesteps $TIMESTEPS \
    $CKPT_FLAG \
    $TARGET_FLAGS
"

echo "==== DONE $(date)"
du -sh "$IMAGE_CACHE" 2>/dev/null || true
du -sh "$TARGET_CACHE" 2>/dev/null || true
