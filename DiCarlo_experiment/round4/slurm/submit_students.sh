#!/bin/bash
# Submit the round 4 students, then the evaluations that depend on them:
#   - one training job per config
#   - one evaluation job PER SEED (control + all conditions of that seed,
#     paired), starting when all of that seed's training jobs have ENDED
#     (afterany: burst sometimes fails a node start and requeues the job;
#     afterok then cancels the evaluation although training succeeded on the
#     retry. evaluate.sbatch itself refuses runs without a finished training.)
#   - one evaluation of the controls of all seeds together (--unpaired):
#     their CKA to each other = how similar two independently seeded
#     students are anyway (seed-to-seed baseline)
#
# Usage (run from anywhere, NOT with sbatch):
#   bash .../submit_students.sh round4/configs/students_target_x8_table1/*.yaml       (all 18)
#   bash .../submit_students.sh round4/configs/students_target_x8_table1/control_s2723.yaml ...   (a subset)
#   NO_EVAL=1 bash .../submit_students.sh ...                                   (training only)
# Config paths are relative to DiCarlo_experiment/. Students of target X run in
# runs/round4/students_X/, their evaluations go to runs/round4/eval_X/.
#
# Before: target cache complete (incl. IT t2/t6/t12), init files made
# (make_inits.sbatch), code committed + pushed + git pull here.

set -euo pipefail

REPO=/scratch/wz2723/dynvision/DynVision
EXP="$REPO/DiCarlo_experiment"
SLURM_DIR="$EXP/round4/slurm"

cd "$EXP"
if [ "$#" -lt 1 ]; then
    echo "usage: bash submit_students.sh <config.yaml> [...]   e.g. round4/configs/students_target_x8_table1/*.yaml"
    echo "config folders:"; ls -d round4/configs/students_* 2>/dev/null
    exit 1
fi
CONFIGS=("$@")

value() {  # value <key> <yaml>: first "key: value" in the file, quotes stripped
    grep -E "^[[:space:]]*$1:" "$2" | head -1 | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]*)"?.*$/\1/' | xargs
}

# ---- checks before anything is submitted ----
declare -A SEED_OF RUNDIR_OF
for CONFIG in "${CONFIGS[@]}"; do
    [ -f "$CONFIG" ] || { echo "ERROR: config $CONFIG not found."; exit 1; }
    RUN_DIR=$(value output_dir "$CONFIG")
    INIT=$(value init_checkpoint "$CONFIG")
    TARGET_DIR=$(value target_dir "$CONFIG")
    [ -e "$RUN_DIR" ] && { echo "ERROR: $RUN_DIR already exists ($CONFIG)."; exit 1; }
    [ -f "$INIT" ] || { echo "ERROR: init file $INIT missing ($CONFIG). Run make_inits.sbatch first."; exit 1; }
    [ -f "$TARGET_DIR/manifest.json" ] || { echo "ERROR: target cache $TARGET_DIR missing ($CONFIG)."; exit 1; }
    STUDENTS_DIR=$(dirname "$RUN_DIR")            # .../runs/round4/students_<target>
    THIS_EVAL="$(dirname "$STUDENTS_DIR")/eval_$(basename "$STUDENTS_DIR" | sed 's/^students_//')"
    if [ -z "${EVAL_ROOT:-}" ]; then EVAL_ROOT="$THIS_EVAL"; fi
    [ "$THIS_EVAL" = "$EVAL_ROOT" ] || { echo "ERROR: configs of different targets in one call ($CONFIG)."; exit 1; }
    SEED_OF[$CONFIG]=$(value init_seed "$CONFIG")
    RUNDIR_OF[$CONFIG]=$RUN_DIR
done
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: repo has uncommitted changes (the jobs will run them):"
    git -C "$REPO" status --short --untracked-files=no
fi
mkdir -p /scratch/wz2723/dynvision/runs/slurm_logs
echo "evaluations will go to: $EVAL_ROOT"

# ---- training ----
declare -A IDS_OF_SEED DIRS_OF_SEED
CONTROL_IDS=(); CONTROL_DIRS=()
for CONFIG in "${CONFIGS[@]}"; do
    ID=$(sbatch --parsable "$SLURM_DIR/train_student.sbatch" "$CONFIG")
    S=${SEED_OF[$CONFIG]}
    IDS_OF_SEED[$S]="${IDS_OF_SEED[$S]:+${IDS_OF_SEED[$S]}:}$ID"
    DIRS_OF_SEED[$S]="${DIRS_OF_SEED[$S]:-} ${RUNDIR_OF[$CONFIG]}"
    if [ "$(value role "$CONFIG")" = "control" ]; then
        CONTROL_IDS+=("$ID"); CONTROL_DIRS+=("${RUNDIR_OF[$CONFIG]}")
    fi
    echo "train $ID  <- $CONFIG"
done

# ---- evaluation (after training) ----
if [ -z "${NO_EVAL:-}" ]; then
    mkdir -p "$EVAL_ROOT"
    for S in $(printf "%s\n" "${!IDS_OF_SEED[@]}" | sort -n); do
        # shellcheck disable=SC2086
        ID=$(sbatch --parsable --dependency=afterany:"${IDS_OF_SEED[$S]}" \
             "$SLURM_DIR/evaluate.sbatch" "$EVAL_ROOT/seed$S" ${DIRS_OF_SEED[$S]})
        echo "eval  $ID  seed $S  (after its training jobs end)"
    done
    if [ "${#CONTROL_IDS[@]}" -ge 2 ]; then
        DEP=$(IFS=:; echo "${CONTROL_IDS[*]}")
        ID=$(sbatch --parsable --dependency=afterany:"$DEP" --export=ALL,EVAL_FLAGS=--unpaired \
             "$SLURM_DIR/evaluate.sbatch" "$EVAL_ROOT/controls_across_seeds" "${CONTROL_DIRS[@]}")
        echo "eval  $ID  controls across seeds (--unpaired)"
    fi
fi
echo "commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo "?") | status: squeue -u ${USER:-$(whoami)}"
