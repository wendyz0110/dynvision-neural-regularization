# Round 3

Fine-tune CordsNet-R8 on monkey N (Round 1 split) with two classification heads and CKA alignment:

    loss = 1 * CE_ImageNet + 1 * CE_THINGS + sum over targets of weight * mean_k log(max(1 - CKA_S, 1e-6))

- **CE_ImageNet**: ImageNet train images, true labels, CordsNet's augmentation (RandomResizedCrop, flip, RandAugment), ImageNet head (logits 0-999); one ImageNet batch per THINGS batch (same size).
- **CE_THINGS**: THINGS images of the 1306 concepts that do not overlap ImageNet, true labels, THINGS head (logits 1000-2305), initialized from `things_head/` (wd 1, epoch 40, lowest val CE). Overlapping concepts: no CE, CKA only.
- **CKA**: as Rounds 1-2 (V1-layer2 bins 2-19, V4-layer6 5-19, IT-layer8 7-19), one recording day per batch.
- Each step: ImageNet batch backward, THINGS batch backward, one AdamW step (gradients add up; one graph in memory at a time).
- AdamW wd 0.01, lr 3e-5 cosine to 0, 20 epochs, batch 128 (to be confirmed by the memtest), seed 2723, fp32.

Runs: `control` (target weights 0), `multi3_w1each` (1 per area), `multi3_w0p33each` (1/3 per area).

## Files
- `lib/model.py`: two-head model, one `weight_norm(Linear(512, 2306))` = [ImageNet head; THINGS head] (per-row weights, so exactly two separate heads).
- `lib/imagenet_data.py`: ImageNet train loader; per epoch a permutation seeded by (seed, epoch), same in every run.
- `lib/things_labels.py`: cache row -> THINGS class (-1 = overlapping concept).
- `lib/evaluate3.py`: epoch evaluation: CKA etc. with `Round1/core/evaluate.py` (ImageNet head only, batch 64), THINGS head on all non-overlapping val images (epoch 0 reproduces 0.630 / 0.835 / 1.696).
- `train/train.py`, `train/memtest.py`; `configs/*.yaml`; `slurm/train.sbatch`, `slurm/memtest.sbatch`, `slurm/submit_round3.sh`.
- `imagenet/eval_imagenet.py --round3_runs ... --epochs 5 10 15 20`: ImageNet-val accuracy of the saved checkpoints (after training).
- Shared code changed backward-compatibly: `Round1/core/evaluate.py` (`out_slice`), `Round1/core/recording.py` (`enabled`); Round 1-2 paths reproduce exactly (checked, differences 0).

Earlier Round 3 analyses (monkey N vs F CKA, ImageNet-val of Rounds 1-2, THINGS head) are in `monkey_cka/`, `imagenet/`, `things_head/`, `analysis/`.
