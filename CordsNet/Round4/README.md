# Round 4

Fine-tune CordsNet-R8 (monkey N) with CE_ImageNet + CE_THINGS + CKA_S alignment, as Round 3, with:

- **THINGS head over all 1854 concepts** (548 that overlap ImageNet + 1306 others), so every THINGS
  image gets a CE. Trained on frozen pretrained features (`things_head/`), initialized from the
  ImageNet head (overlapping concepts; one matched class: copied, several: mean) and the Round 3
  head (the others). Selected: all rows trained, weight decay 0.1, epoch 21 (val top-1 0.630).
- **Every THINGS image used every epoch**: per day, images shuffled and cut into batches; a day's
  last batch is filled with other images of the same day instead of dropped (`core4/batches.py`).
  Same for the val images in the evaluation.
- **Batch 96** (ImageNet and THINGS).

Everything else as Round 3: three configs (control, 3 areas x 1, 3 areas x 1/3), CKA targets and
windows, AdamW lr 3e-5 cosine to 0, wd 0.01, 20 epochs, seed 2723, CordsNet's ImageNet augmentation.

## Model

`CordsNet/common/cordsnet_things.py`: `build_cordsnet_things(checkpoint, things_head)` returns
DynVision's CordsNet-R8 (authors' checkpoint) whose last layer has 1000 + 1854 outputs, plus
`heads` (output slices, THINGS class order, `heads.labels(concepts)`). Only the last layer is
replaced, so DynVision's time steps and response recording work as for the original model. The
THINGS head is one self-contained file (weights, class names, overlap with ImageNet, description),
made by `things_head/export_head.py`.

## Steps (on Torch)

| step | command (from `CordsNet/...`) | output |
|---|---|---|
| 1. features (done) | `Round3/slurm`: `sbatch things_features.sbatch --concepts all --out_dir CordsNet/Round4/outputs/things_head` | `outputs/things_head/features.npz` |
| 2. THINGS head (done) | `Round4/slurm`: `sbatch things_head.sbatch` | `outputs/things_head/cosine_ep40/` |
| 3. head file | `Round4/slurm`: `sbatch export_head.sbatch` | `outputs/things_head/things_head_1854.pt` |
| 4. memory test | `Round4/slurm`: `sbatch memtest.sbatch` | `outputs/memtest/memtest_<job>.json` |
| 5. pilot | `Round4/slurm`: `sbatch train.sbatch ../configs/multi3_w1each.yaml --epochs 1 --max_steps 20 --output_dir <dir>` | |
| 6. training | `bash Round4/slurm/submit_round4.sh` (3 runs, each chained with afterany) | `outputs/<config>/` |

Batch 96 needs an H200 (Round 3 memtest: 141.7 GB peak), so steps 4-6 use `--partition=h200_cds,h200_public`.

## Outputs per run

`val_metrics.csv` (epoch 0 and every epoch: THINGS top-1 / top-5 accuracy at 200 ms and weighted
CE on every val image; val CKA_S of each target), `train_metrics.csv` (every step: both CEs and
top-1 accuracies, CKA loss and per-bin CKA_S), `eval/epoch_XXX.npz`, `checkpoints/epoch_XXX.pt`
(epochs 5, 10, 15, 20), `model_card.yaml`. ImageNet-val accuracy: evaluated afterwards on the
saved checkpoints.
