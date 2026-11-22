#!/bin/bash
# Submit the three Round 3 runs (control, 3 areas x 1, 3 areas x 1/3), one GPU job each, in parallel.
# Each run is chained N_JOBS times with afterany: every job resumes from the run's last finished
# epoch (checkpoints/last.pt) and exits at once if all epochs are done.
#
#   bash /scratch/wz2723/dynvision/DynVision/CordsNet/Round3/slurm/submit_round3.sh
#   PARTITION=h200_public N_JOBS=3 bash .../submit_round3.sh
set -euo pipefail
SLURM_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SLURM_DIR"
PARTITION=${PARTITION:-h200_cds}
N_JOBS=${N_JOBS:-2}

for NAME in control multi3_w1each multi3_w0p33each; do
    CONFIG="../configs/${NAME}.yaml"                  # relative to Round3/train, where train.py runs
    [ -f "$SLURM_DIR/$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
    DEP=""
    for i in $(seq 1 "$N_JOBS"); do
        JOB=$(sbatch --parsable --partition="$PARTITION" --job-name="r3_${NAME}" $DEP train.sbatch "$CONFIG")
        echo "$NAME: job $JOB ($i of $N_JOBS) on $PARTITION"
        DEP="--dependency=afterany:${JOB}"
    done
done
