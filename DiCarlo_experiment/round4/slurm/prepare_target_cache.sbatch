#!/bin/bash
# Build the TARGET cache for a trained target (round4/prepare_targets.py =
# round 2's script, but the target is rebuilt from its model card, so any
# architecture works): logits at all 20 timesteps for every image of the
# image cache (train, val AND test), V1/V4/IT activations at the last
# timestep, and (second pass, full mode) IT at t2/t6/t12 for the
# multi-timestep condition and the monitored CKA. One job does both passes.
# The students read their pseudo-labels and CKA targets from this folder.
#
# The image cache already exists and is reused (it does not depend on the target).
# The target's model_card.yaml is copied into the cache folder, so the cache
# records which model produced it.
#
# Cluster: NYU burst, account gm3239, partition c12m85-a100-1.
#
# Usage (from the repo root, /scratch/wz2723/dynvision/DynVision):
#   sbatch DiCarlo_experiment/round4/slurm/prepare_target_cache.sbatch dry_run
#   sbatch DiCarlo_experiment/round4/slurm/prepare_target_cache.sbatch full
# Optional 2nd / 3rd arguments override the target run folder and the cache folder:
#   sbatch ... full <target run dir> <target cache dir>
#
# Which activations to store: V1, V4, IT at the last timestep, then IT at
# t2/t6/t12 (EXTRA_IT_TIMESTEPS, "" = skip) by default.
# To ADD other ones to an existing cache (existing files are kept, compatibility is
# checked), set CACHE_LAYERS / CACHE_TIMESTEPS (timesteps separated by ":"):
#   sbatch --export=ALL,CACHE_LAYERS=IT,CACHE_TIMESTEPS=2:6:12 \
#          DiCarlo_experiment/round4/slurm/prepare_target_cache.sbatch full
#
#   dry_run: runs a few batches, prints shapes, projected size and time, writes nothing
#   full:    writes the cache
#
#SBATCH --job-name=r4_targetcache
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

MODE=${1:-dry_run}
TARGET_RUN=${2:-/scratch/wz2723/dynvision/runs/round4/target_x8_aug}
TARGET_CACHE=${3:-/scratch/wz2723/dynvision/cache/targets/target_x8_aug}

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
IMAGE_CACHE=/scratch/wz2723/dynvision/cache/imagenette_images

LAYERS=${CACHE_LAYERS:-"V1 V4 IT"}
TIMESTEPS=${CACHE_TIMESTEPS:-"-1"}
# Second pass (only when CACHE_LAYERS / CACHE_TIMESTEPS are not set):
EXTRA_IT_TIMESTEPS=${EXTRA_IT_TIMESTEPS-"2 6 12"}
if [ -n "${CACHE_LAYERS:-}${CACHE_TIMESTEPS:-}" ]; then EXTRA_IT_TIMESTEPS=""; fi
LAYERS=${LAYERS//:/ }
TIMESTEPS=${TIMESTEPS//:/ }

case "$MODE" in
    dry_run) TARGET_FLAGS="--dry_run" ;;
    full)    TARGET_FLAGS="" ;;
    *) echo "ERROR: mode must be 'dry_run' or 'full', got '$MODE'"; exit 1 ;;
esac

CARD="$TARGET_RUN/model_card.yaml"
if [ ! -f "$CARD" ]; then
    echo "ERROR: $CARD not found (is the target trained?)"; exit 1
fi
if [ ! -f "$IMAGE_CACHE/manifest.json" ]; then
    echo "ERROR: image cache $IMAGE_CACHE not found"; exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date) | mode: $MODE"
echo "target run: $TARGET_RUN (checkpoint: the one its model card names)"
echo "image cache: $IMAGE_CACHE | target cache: $TARGET_CACHE"
echo "layers: $LAYERS | timesteps: $TIMESTEPS | then IT at: ${EXTRA_IT_TIMESTEPS:-none}"
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
python -u round4/prepare_targets.py \
    --image_cache '$IMAGE_CACHE' \
    --target_run '$TARGET_RUN' \
    --out_dir '$TARGET_CACHE' \
    --layers $LAYERS \
    --timesteps $TIMESTEPS \
    $TARGET_FLAGS
if [ '$MODE' = full ] && [ -n '$EXTRA_IT_TIMESTEPS' ]; then
    python -u round4/prepare_targets.py \
        --image_cache '$IMAGE_CACHE' \
        --target_run '$TARGET_RUN' \
        --out_dir '$TARGET_CACHE' \
        --layers IT \
        --timesteps $EXTRA_IT_TIMESTEPS
fi
"

if [ "$MODE" = "full" ]; then
    du -sh "$TARGET_CACHE"    # target_model_card.yaml is copied in by prepare_targets.py
fi
echo "==== DONE $(date)"
