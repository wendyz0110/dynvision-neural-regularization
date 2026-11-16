# Round 2: aligning three areas at once (monkey N)

V1 vs layer2, V4 vs layer6, IT vs layer8, CKA_S loss, everything else as Round 1
(batch 64 from one recording day, 10 epochs, lr 3e-5 cosine, AdamW, fp32, CE on pseudo-labels, seed 2723).

| run | V1-layer2 | V4-layer6 | IT-layer8 | sum |
|---|---|---|---|---|
| `multi3_w1each` | 1.0 | 1.0 | 1.0 | 3 |
| `multi3_w0p33each` | 1/3 | 1/3 | 1/3 | 1 |

CKA bins per area: TVSD window start (bin containing it) to 200 ms: V1 bins 2-19, V4 5-19, IT 7-19.
Layers two apart (layer2, layer6, layer8) so that the V4 target is not mixed with the spread of the IT
target to layer7 seen in Round 1.

Comparisons: Round 1 `control` (CE only) and `student_IT_layer8_w1` (IT only, weight 1) are reused;
same code (checked: the updated code reproduces Round 1 exactly), data, seed and batches.

## Code

The training code is shared with Round 1 and lives there:
- `CordsNet/Round1/train/train.py`, `memtest.py`, `slurm/train.sbatch`
- `CordsNet/Round1/core/` (data, recording, losses incl. `parse_targets`, evaluation, accuracy)

This folder holds only Round 2's configs, job scripts, outputs and analysis:
- `configs/` run configs (`loss.targets`)
- `slurm/memtest_round2.sbatch`, `slurm/submit_round2.sh`
- `outputs/<run>/` same files as Round 1 (`train_metrics.csv`, `val_metrics.csv` with CKA_S per target and
  accuracy, `eval/`, `checkpoints/`, `model_card.yaml`); not in git
