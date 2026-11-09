#!/bin/bash
# Submit the Round 1 student and control, each as a chain of N_JOBS jobs that resume from
# the last finished epoch (afterany: the next job starts whether the previous one finished
# or hit its time limit; a job whose run is already complete exits immediately).
#   bash submit_round1.sh            # 3 jobs per run
#   N_JOBS=2 PARTITION=h200_public bash submit_round1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
N_JOBS=${N_JOBS:-3}
PART=${PARTITION:+--partition=$PARTITION}

for CONFIG in configs/student_IT_layer8_w0p1.yaml configs/control.yaml; do
    NAME=$(basename "$CONFIG" .yaml)
    PREV=$(sbatch --parsable $PART --job-name="r1_${NAME}" slurm/train.sbatch "$CONFIG")
    echo "$NAME: job $PREV"
    for i in $(seq 2 "$N_JOBS"); do
        PREV=$(sbatch --parsable $PART --job-name="r1_${NAME}" --dependency=afterany:"$PREV" slurm/train.sbatch "$CONFIG")
        echo "$NAME: job $PREV (after the previous one)"
    done
done
