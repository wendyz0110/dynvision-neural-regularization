# Round 4: DyRCNNx8 target, DyRCNNx4 students

Same design as round 3 (6 conditions x 3 seeds, same init files, same data
order, same recipe). Only the target changes: a DyRCNNx8 with the same time
constant, delays, recurrence type, skip and feedback settings and the same
training recipe as round 3's DyRCNNx4 `target_r3`.

Two x8 targets:
- `target_x8` (first attempt, no augmentation): memorized the training images
  within ~8 epochs (best val CE at epochs_completed 8, train CE ~0.001 after).
  Kept for reference, not used.
- `target_x8_aug` (used): the same plus DynVision's Imagenette training
  augmentation on the training batches only (rotation 10 deg, translate 10%,
  horizontal flip, colour jitter 0.2). Validation, the target cache and the
  students use un-augmented images; the students are not augmented (their
  targets come from a cache of un-augmented images).
- `target_x8_table1` (in preparation): DyRCNNx8 with the DynVision paper's
  default training configuration (Table 1): feedback off, Adam lr 8e-4 wd 5e-4,
  effective batch 1024 (64 x 16), cosine T_max 250, clip 1.0, CE over all
  timesteps, plus an activity loss (weight 0.0002 as in Tiannan's config) and
  the same augmentation. Differences from Table 1 are listed at the top of
  `configs/target_x8_table1.yaml`.

New in round 4 (kept for later rounds):
- `augmentation.transforms` in the target YAML (training batches only).
- `loss.activity_weight` / `activity_modules` / `activity_ord`: activity loss with
  DynVision's ActivityLoss formula on the named modules (default `tstep_*`).
- Training measures the initial weights before the first epoch, with the same
  code as every epoch. That measurement is the first row of `metrics/metrics.csv`
  (`epoch` = -1, `epochs_completed` = 0). Plot against `epochs_completed`.
- `train_target.load_trained_target()` rebuilds any trained target from its
  `model_card.yaml`. The cache script and the evaluation use it, so the target
  architecture is no longer hard-coded.
- The evaluation handles a target and students with different feature sizes.

## Steps on burst (from `DiCarlo_experiment/`)

1. `sbatch round4/slurm/train_target.sbatch round4/configs/target_x8_aug.yaml`
   - Check `s` per epoch and GPU memory in the log after the first epochs.
   - If the job hits its time limit, resubmit with `resume`.
2. Check the target: curves, val/test accuracy, and train vs val (overfitting).
3. `sbatch round4/slurm/prepare_target_cache.sbatch dry_run`, then
   `sbatch round4/slurm/prepare_target_cache.sbatch full`.
   - The full job writes V1/V4/IT at t19, then IT at t2/t6/t12.
   - Expected size is about 23 GB.
4. `bash /scratch/wz2723/dynvision/DynVision/DiCarlo_experiment/round4/slurm/submit_students.sh`
   - Submits 18 trainings, then 3 per-seed evaluations and the controls-across-seeds evaluation.
   - The init files are round 3's (`cache/student_init/x4_seed{2723,2724,2725}.pt`).

Outputs:
- `runs/round4/target_x8_aug` (and the unused `runs/round4/target_x8`)
- `cache/targets/target_x8_aug`
- `runs/round4/students/<condition>_s<seed>`
- `runs/round4/eval/`
