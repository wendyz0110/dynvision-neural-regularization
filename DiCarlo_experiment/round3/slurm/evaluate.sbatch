#!/bin/bash
# Evaluate finished round 3 student runs together with the target on the test
# set (round3/evaluate_alignment.py). submit_students.sh queues one of these
# per seed automatically (after that seed's training jobs succeed). By hand:
#
#   sbatch DiCarlo_experiment/round3/slurm/evaluate.sbatch <out_dir> <run_dir> [<run_dir> ...]
#
# e.g. one seed (paired: same init file):
#   sbatch ... /scratch/wz2723/dynvision/runs/round3/eval/seed2723 \
#              /scratch/wz2723/dynvision/runs/round3/students/control_s2723 \
#              /scratch/wz2723/dynvision/runs/round3/students/IT_w1_s2723 ...
# Runs with different init files (e.g. the three controls, seed-to-seed
# baseline) need --unpaired:
#   sbatch --export=ALL,EVAL_FLAGS=--unpaired ... <out_dir> <run_dir> ...
#
# Expected time: round 2 took 85 min for target + 4 students (125 passes);
# target + 6 students = 175 passes -> ~2 h.
#
#SBATCH --job-name=r3_eval
#SBATCH --account=gm3239
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/wz2723/dynvision/runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

[ "$#" -ge 2 ] || { echo "usage: sbatch evaluate.sbatch <out_dir> <run_dir> [<run_dir> ...]"; exit 1; }
OUT_DIR=$1; shift
if [ -e "$OUT_DIR" ]; then
    echo "ERROR: $OUT_DIR already exists (results are never overwritten)."; exit 1
fi
for RUN_DIR in "$@"; do
    if [ ! -f "$RUN_DIR/checkpoints/last.pt" ] || [ ! -f "$RUN_DIR/training_summary.json" ]; then
        echo "ERROR: $RUN_DIR has no last.pt / training_summary.json (training not finished?)."
        exit 1
    fi
done
RUNS=$(printf "'%s' " "$@")
EVAL_FLAGS=${EVAL_FLAGS:-}

echo "job $SLURM_JOB_ID on $(hostname) | start $(date)"
echo "out: $OUT_DIR | flags: $EVAL_FLAGS"
echo "runs: $*"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv \
    --overlay "$OVERLAY:ro" \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/DiCarlo_experiment'
python -u round3/evaluate_alignment.py --out_dir '$OUT_DIR' $EVAL_FLAGS --runs $RUNS
"

echo "==== DONE $(date)"
