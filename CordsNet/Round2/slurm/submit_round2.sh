#!/bin/bash
# Submit the two Round 2 runs (V1-layer2, V4-layer6, IT-layer8 aligned at once), one GPU job
# each, at the same time: weight 1.0 per area (sum 3) and 1/3 per area (sum 1).
# Training code: the shared Round1/train/train.py (via Round1/train/slurm/train.sbatch).
# Controls: Round 1 control (CE only) and student w1 (IT only) are reused (same code, data, seed, batches).
# Each job resumes from its last finished epoch, so if one is interrupted, run this script again.
#
#   bash /scratch/wz2723/dynvision/DynVision/CordsNet/Round2/slurm/submit_round2.sh
#   PARTITION=h200_public bash .../submit_round2.sh
set -euo pipefail
ROUND2=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_DIR="$ROUND2/../Round1/train"
cd "$TRAIN_DIR"
PARTITION=${PARTITION:-h200_cds}

for NAME in multi3_w1each multi3_w0p33each; do
    CONFIG="../../Round2/configs/${NAME}.yaml"        # relative to Round1/train, where train.sbatch runs
    [ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
    JOB=$(sbatch --parsable --partition="$PARTITION" --job-name="r2_${NAME}" slurm/train.sbatch "$CONFIG")
    echo "$NAME: job $JOB on $PARTITION"
done
