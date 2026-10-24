#!/bin/bash
# Build the TARGET cache for a trained target (prepare_targets.py, unchanged
# from round 2): logits at all 20 timesteps for every image of the image
# cache (train, val AND test), plus V1/V4/IT activations at the last timestep.
# The students read their pseudo-labels and CKA targets from this folder.
#
# The image cache already exists and is reused (it does not depend on the target).
# The target's model_card.yaml is copied into the cache folder, so the cache
# records which model produced it.
#
# Cluster: NYU burst, account gm3239, partition c12m85-a100-1.
#
# Usage (from the repo root, /scratch/wz2723/dynvision/DynVision):
#   sbatch DiCarlo_experiment/round3/slurm/prepare_target_cache.sbatch dry_run
#   sbatch DiCarlo_experiment/round3/slurm/prepare_target_cache.sbatch full
# Optional 2nd / 3rd arguments override the target run folder and the cache folder:
#   sbatch ... full <target run dir> <target cache dir>
#
#   dry_run: runs a few batches, prints shapes, projected size and time, writes nothing
#   full:    writes the cache
#
#SBATCH --job-name=r3_targetcache
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
TARGET_RUN=${2:-/scratch/wz2723/dynvision/runs/round3/target_r3}
TARGET_CACHE=${3:-/scratch/wz2723/dynvision/cache/targets/target_r3}

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
IMAGE_CACHE=/scratch/wz2723/dynvision/cache/imagenette_images

LAYERS="V1 V4 IT"
TIMESTEPS="-1"

case "$MODE" in
    dry_run) TARGET_FLAGS="--dry_run" ;;
    full)    TARGET_FLAGS="" ;;
    *) echo "ERROR: mode must be 'dry_run' or 'full', got '$MODE'"; exit 1 ;;
esac

# The checkpoint the target's model card names as "the" target (best.pt / last.pt).
CARD="$TARGET_RUN/model_card.yaml"
if [ ! -f "$CARD" ]; then
    echo "ERROR: $CARD not found (is the target trained?)"; exit 1
fi
CKPT_NAME=$(grep -E '^[[:space:]]*target_checkpoint:' "$CARD" | head -1 | sed -E 's/.*:[[:space:]]*//' | xargs)
CHECKPOINT="$TARGET_RUN/checkpoints/$CKPT_NAME"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint $CHECKPOINT not found"; exit 1
fi
if [ ! -f "$IMAGE_CACHE/manifest.json" ]; then
    echo "ERROR: image cache $IMAGE_CACHE not found"; exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname) | start $(date) | mode: $MODE"
echo "target run: $TARGET_RUN | checkpoint: $CHECKPOINT"
echo "image cache: $IMAGE_CACHE | target cache: $TARGET_CACHE"
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
python -u prepare_targets.py \
    --image_cache '$IMAGE_CACHE' \
    --out_dir '$TARGET_CACHE' \
    --checkpoint '$CHECKPOINT' \
    --layers $LAYERS \
    --timesteps $TIMESTEPS \
    $TARGET_FLAGS
"

if [ "$MODE" = "full" ]; then
    cp "$CARD" "$TARGET_CACHE/target_model_card.yaml"
    du -sh "$TARGET_CACHE"
fi
echo "==== DONE $(date)"
