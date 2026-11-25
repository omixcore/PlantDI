# T4 Supervised Domain-Adapted Upper Bound - Final Results (local GPU)

*English translation of the frozen T4 report. Editorial note: this is an ORACLE upper bound and must not be compared directly with zero-shot rows. The deployable non-oracle result is reported in `RESULTS_T4b_E2E_FINAL.md`.*

Date: 2026-11-25. Hardware: RTX 3060 Laptop 6 GB, torch 2.5.1+cu124, SAM2 v1.0 (tiny).

> Bottom line: T4 performed real training and evaluation on the local machine. It **substantially improved pixel-level segmentation** (mIoU 0.108 -> 0.496, Dice 0.154 -> 0.607), and all three representative cases approached Expert GT. However, it uses **oracle box prompts** (bounding boxes derived from the test-set GT masks), so it is an **optimistic upper bound**, not a deployable result, and it is **not directly comparable with the zero-shot rows**. Its MAE is worse than zero-shot because of systematic over-prediction (86/100 images too large).

## 1. Can it be done locally? Yes.

The earlier "CPU-only / infeasible" judgement referred to the remote server. The local machine actually has a GPU:

| Condition | Measured |
|---|---|
| GPU | RTX 3060 Laptop 6 GB, driver 566.07 / CUDA 12.7 |
| CUDA torch | 2.5.1+cu124, `torch.cuda.is_available()=True`, GPU matmul passed |
| SAM2 | facebookresearch/sam2 v1.0 source + `sam2_hiera_tiny.pt` (156 MB, HF mirror) |
| Data | 7,916 official train images (image/mask/bbox), zero-overlap audit passed |
| Training time | 2 epochs x 7,916 images = 3,376 s (about 56 minutes) |

## 2. Isolation and audit (evidence for the Methods section)

- `data/plantseg_train/t4_data_audit.json`: overlap of the training split with the official Test set (2,295 images) and with PlantSeg-100 (100 images) is **zero in both cases**.
- Prompt source: official annotation boxes (xywh -> xyxy), **not** GT masks and **not** VLM; no test-set information was used during training.
- The image encoder and prompt encoder were frozen; only `sam_mask_decoder` was trained (4,215,109 parameters).
- Weight SHA256:
  - Initial `sam2_hiera_tiny.pt` = `65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125`
  - Fine-tuned `sam2_tiny_decoder_ft.pt` = `7a7879fa4ffd235a63c550201d9ce539b20bdba4df01bbd6b26a91c8c5aaa370`
- Loss: epoch 1 mean 0.772 -> epoch 2 mean 0.610.
- Full log: `results/t4_finetune.log`; training script: `scripts/finetune_sam2_decoder.py`.

## 3. Evaluation protocol (must be reported explicitly)

PlantSeg-100 test images have no official box annotations. To **isolate localization error and measure pixel-level mask quality alone**, evaluation used the bounding box of each Expert GT mask as the prompt (labelled `oracle_gt_bbox`). Because test-set ground truth is used:

- every T4 number is an **optimistic upper bound**;
- it **must not** be placed in the same column as v6/v7 zero-shot results as if it were method performance;
- the summary table `benchmark_v7_summary.csv` uses a separate paradigm label and a `note` column.

The control arm `sam2_base_oracle` uses the **same** oracle box with the un-fine-tuned decoder, so the difference between the two rows reflects only the decoder fine-tuning effect at the pixel level.

## 4. Quantitative results (PlantSeg-100, n=100, original scale)

| Method | r | Spearman rho | CCC | MAE | mIoU | Dice |
|---|---|---|---|---|---|---|
| Original SAM2 + oracle box | 0.320 | 0.231 | 0.256 | **0.145** | 0.108 | 0.154 |
| **Fine-tuned SAM2 + oracle box** | **0.545** | **0.557** | **0.366** | 0.197 | **0.496** | **0.607** |
| (reference) v7 zero-shot, non-oracle | 0.543 | 0.512 | 0.468 | 0.110 | 0.247 | 0.360 |

Key points:

- r improves from 0.320 to 0.545 (p=4.7e-9), numerically close to v7 zero-shot 0.543, but **the protocols differ and the values must not be equated**.
- **The pixel-level gain is the real value of T4**: mIoU improves 4.6-fold and Dice nearly 4-fold, a level never reached by the zero-shot path.
- **Limitation**: MAE rises from 0.145 to 0.197; the fine-tuned decoder tends to fill the oracle box, over-predicting 86 of 100 images (mean signed bias +0.187). This tension between pixel-boundary fidelity and DI quantification is a target for follow-up work.

## 5. Three representative cases (fine-tuned SAM2 + oracle box)

| Case | Expert GT | Original SAM2 | Fine-tuned SAM2 | Fine-tuned IoU / Dice |
|---|---|---|---|---|
| Case 3 bean mosaic virus 0098 | 37.4% | 1.9% | **37.0%** | 0.95 / 0.98 |
| Case 4 tobacco mosaic virus 117 | 82.1% | 7.4% | **71.9%** | 0.86 / 0.92 |
| Case 5 basil downy mildew 0067 | 19.5% | 3.5% | **29.9%** | 0.64 / 0.78 |

The overlays are in `figures/qual_overlays_t4/`; caption numbers come from the frozen detail JSON, not from a fresh re-run.

## 6. Deliverables

- Training: `scripts/finetune_sam2_decoder.py`, `data/plantseg_train/*`, `weights/sam2_tiny_decoder_ft.pt` + `_audit.json`, `results/t4_finetune.log`.
- Evaluation: `scripts/eval_t4_upperbound.py`, `results/t4_upperbound_summary.csv`, `results/t4_upperbound_detail.json`.
- Summary: `results/benchmark_v7_summary.csv` (15 rows at the time of this snapshot, with two T4 rows marked COMPUTED and given a separate paradigm + note).
- Figures: `figures/fig_v7_t4_benchmark.{svg,pdf}`, `figures/qual_overlays_t4/*.png` (3 images, 300 dpi).
- Environment: CUDA torch wheels in `data/_download/wheels/`; SAM2 source in `third_party/sam2/`; no historical directory was modified.

## 7. Honest boundary

T4 shows that, **given correct localization**, supervised fine-tuning can bring pixel-level masks close to expert ground truth. It does **not** solve the localization failure in zero-shot v7 Case 3 (the VLM failed to box the lesion), because the oracle box skips localization entirely. If the paper reports deployable performance, the main result must remain zero-shot v7 (r=0.543); T4 belongs in a separate "Supervised Domain-Adapted Upper Bound" row with an explicit oracle protocol.
