#!/bin/bash
# Monkey F: split CSV + training cache, from the corrected data (V4/IT swapped back, monkeyF_clean).
#  1. make_split.py with monkey N's seed: same train/val/test per image as monkey N, F's own day/day_row
#  2. check that F's split equals N's (stim_idx, concept, split)
#  3. build_cache.py -> /scratch/wz2723/data/TVSD/round1_cache_monkeyF (images, F neural z-scored per day, meta)
# CPU only, ~5-10 min.
#   sbatch /scratch/wz2723/dynvision/DynVision/CordsNet/Round1/prepare/slurm/build_cache_monkeyF.sbatch

#SBATCH --job-name=cache_monkeyF
#SBATCH --account=torch_pr_628_cds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2723/cordsnet_runs/slurm_logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2723@nyu.edu

set -euo pipefail
REPO=/scratch/wz2723/dynvision/DynVision
OVERLAY=/scratch/wz2723/dynvision/overlay-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

echo "==== START $(date)  job ${SLURM_JOB_ID} on $(hostname)  code $(git -C "$REPO" rev-parse --short HEAD)"
singularity exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate dynvision
cd '$REPO/CordsNet/Round1'

python -u make_split.py \
    --neural_dir /scratch/wz2723/data/TVSD/monkeyF_clean \
    --monkey monkeyF \
    --manifest /scratch/wz2723/data/THINGS/things_manifest_tvsd.csv \
    --out splits/monkeyF_split.csv

python -u - <<'EOF'
import csv
from collections import Counter
n = list(csv.DictReader(open('splits/monkeyN_split.csv')))
f = list(csv.DictReader(open('splits/monkeyF_split.csv')))
cols = ['stim_idx', 'concept', 'things_path', 'split']
assert len(n) == len(f) and all(a[c] == b[c] for a, b in zip(n, f) for c in cols), 'monkey F split differs from monkey N split'
print('check: monkey F split == monkey N split (same train/val/test per image)')
print('images per (N day, F day):', sorted(Counter((a['day'], b['day']) for a, b in zip(n, f)).items()))
EOF
cd prepare
python -u build_cache.py \
    --split_csv ../splits/monkeyF_split.csv \
    --manifest /scratch/wz2723/data/THINGS/things_manifest_tvsd.csv \
    --neural_dir /scratch/wz2723/data/TVSD/monkeyF_clean \
    --monkey monkeyF \
    --pseudo_labels /scratch/wz2723/cordsnet_runs/inference/things_cordsnetr8/outputs.npz \
    --out_dir /scratch/wz2723/data/TVSD/round1_cache_monkeyF \
    --num_workers ${SLURM_CPUS_PER_TASK}
"
echo "==== DONE $(date)"
