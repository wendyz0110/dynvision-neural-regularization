#!/bin/bash
# Submit the three Round 1 runs (control, student w_cka 0.1, student w_cka 1.0), one GPU job
# each, all at the same time. fp32 with batch 64 needs ~93 GB, so they go to an H200 partition.
# A full run (10 epochs) takes ~2 h; each job resumes from its last finished epoch, so if a
# job is interrupted, simply run this script again (finished runs exit immediately).
#
#   bash submit_round1.sh                          # H200 partition h200_cds
#   PARTITION=h200_public bash submit_round1.sh    # other H200 partition
set -euo pipefail
cd "$(dirname "$0")/.."
PARTITION=${PARTITION:-h200_cds}

for CONFIG in configs/control.yaml configs/student_IT_layer8_w0p1.yaml configs/student_IT_layer8_w1.yaml; do
    NAME=$(basename "$CONFIG" .yaml)
    JOB=$(sbatch --parsable --partition="$PARTITION" --job-name="r1_${NAME}" slurm/train.sbatch "$CONFIG")
    echo "$NAME: job $JOB on $PARTITION"
done
