# T4b Final Results: Real Non-Oracle End-to-End Comparison (Cached VLM Prompt + FT-SAM2)

Date: 2026-11-25. Hardware: local RTX 3060 6 GB (CUDA). Samples: PlantSeg-100, n=100, original scale.

---

## 0. Protocol and reproducibility audit

- **Prompt source**: the frozen v7 image-only prompts (`POLYGON_PRIMARY_V7` -> `POLYGON_RETRY_V7`); only original image bytes were sent. **No** filename, crop/disease label, GT mask or other metadata was sent.
- **Call strategy**: one effective localization per image (primary, with forced-recall retry only if primary produced no polygon/point); **no TTA**, temperature=0, effort=high, max_tokens=4096; 4 workers with per-image checkpointing.
- **Cache**: `results/t4b_vlm_prompts_cache.json`, 100/100 records; SHA256 `782fcc602677bacca9aa24530c57af3906baddbba831ad5c51091a09778c39c2`. Sources: 77 primary / 23 retry; 21 images with point prompts; mean 6.98 boxes/image (median 6, max 12); no image without a box. Tokens: 81,003 input / 283,449 output; summed per-image latency 1,793.8 s, wall clock 452.4 s with 4 workers. The key was read only from a process environment variable; the cache stores only its fingerprint `6ea770ca`.
- **Weight SHA256**: base `sam2_hiera_tiny.pt` = `65b50056...d1125`; FT `sam2_tiny_decoder_ft.pt` = `7a7879fa...aaa370` (identical to T4).
- **GT scope**: the GT mask was used **only** for final scoring and error attribution; every prompt came from the cache above, and no box or point was generated from `data/plantseg100/masks/`.
- **Strict paired protocol**: Arms 1-4 all consume the same cache; all 100 binary predictions are stored per image in `results/t4b_masks/*.npz` (gt / leaf / four arms), so the numbers and the figures are exactly reproducible.
- Logs: `results/t4b_vlm_fetch.log`, `results/t4b_e2e_eval.log`; audit: `results/t4b_e2e_summary_audit.json`.

---

## 1. Four paired arms (same non-oracle cache, n=100)

| Arm | Method | r (p) | Spearman rho | CCC | MAE | mIoU | Dice | mean(pred-gt) | n_over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `t4b_paired_vlm_only` (VLM polygons + leaf clipping, no SAM2) | 0.3678 (1.67e-4) | 0.4355 | 0.2398 | 0.1254 | 0.2049 | 0.3056 | -0.1082 | 19/100 |
| 2 | `t4b_e2e_raw_sam2` (cached boxes/points + zero-shot SAM2) | 0.3285 (8.49e-4) | 0.4369 | 0.2371 | 0.1165 | 0.2713 | 0.3758 | -0.0927 | 20/100 |
| 3 | `t4b_e2e_ft_sam2` (cached boxes/points + FT-SAM2) | **0.5756 (3.73e-10)** | **0.6788** | **0.5457** | **0.1083** | **0.5579** | **0.6806** | +0.0418 | 58/100 |
| 4 | `t4b_e2e_ft_sam2_leafclip` (Arm 3 AND leaf mask) | 0.5291 (1.52e-8) | 0.5940 | 0.5170 | 0.1137 | 0.4288 | 0.5455 | -0.0080 | 35/100 |

Pure GPU SAM2 inference speed: Arm 2 = 0.305 s/image, Arms 3/4 = 0.218 s/image (excluding VLM). Leaf clipping reduces the systematic over-prediction from +0.0418 to -0.0080 (over-predicted images 58 -> 35), at the cost of mIoU -0.1291 and Dice -0.1351: it mainly corrects area bias rather than pixel alignment.

---

## 2. Question 1: does FT-SAM2 improve over Pure VLM and v7 Zero-Shot, or trade off?

Using Arm 3 as the deployable result:

| Baseline | r | MAE | mIoU | Dice | CCC |
|---|---:|---:|---:|---:|---:|
| Pure VLM (frozen benchmark) | 0.3316 | 0.1280 | 0.2576 | 0.3814 | 0.3122 |
| v7 Zero-Shot (frozen benchmark) | 0.5434 | 0.1101 | 0.2475 | 0.3602 | 0.4677 |
| **T4b Arm 3** | **0.5756** | **0.1083** | **0.5579** | **0.6806** | **0.5457** |
| Arm 3 - Pure VLM | **+0.2440** | **-0.0196** | **+0.3004** | **+0.2991** | +0.2335 |
| Arm 3 - v7 Zero-Shot | **+0.0322** | **-0.0018** | **+0.3105** | **+0.3204** | +0.0780 |

**Conclusion: all four dimensions improve; no trade-off was observed.** Relative to Pure VLM, correlation improves by 0.244 and mIoU by 0.300. Relative to v7 Zero-Shot, correlation improves slightly (+0.032) and pixel-level mIoU/Dice improve by about 0.31/0.32, while MAE decreases only marginally (v7's TTA and training-quantile calibration had already pushed MAE very low).

Caveat: the Pure VLM / v7 numbers come from the frozen `benchmark_v7_summary.csv`; v7 used TTA double calls and calibration, while T4b is a new single-view localization batch. **Only Arms 1-4 are strictly paired**; cross-row comparisons are cross-run comparisons, not same-prompt ablations.

---

## 3. Question 2: T4 Oracle vs T4b Non-Oracle, and bottleneck shares

### 3.1 Direct comparison with T4 Oracle

| Metric | T4 Oracle (single GT box) | T4b Arm 3 (non-oracle multi-box) | Difference |
|---|---:|---:|---:|
| r | 0.5445 | 0.5756 | +0.0311 |
| MAE | 0.1975 | 0.1083 | -0.0891 |
| mIoU | 0.4963 | 0.5579 | **+0.0616** |
| Dice | 0.6073 | 0.6806 | +0.0733 |
| mean bias | +0.1872 (86/100 over) | +0.0418 (58/100 over) | strongly reduced |

**It must be stated honestly that non-oracle T4b outperforms T4 Oracle on mIoU/Dice/MAE; this cannot be read as "VLM localization is more accurate than GT."** The protocols differ: T4 Oracle used **one GT-box prompt plus a single mask**, whereas T4b uses on average **6.98 VLM lesion boxes plus multimask self-scoring and union**. The multi-box union covers more true lesions and strongly reduces the over-filling caused by a single enclosing box (86/100 -> 58/100 over-predicted). Therefore T4's 0.496 is a pixel upper bound under a single-box protocol, not an absolute upper bound over all oracle protocols.

### 3.2 Error attribution (GT used only for scoring, never for prompting)

Decompose per-image IoU error `(FN + FP)/|GT union pred|` into:
- `loc-limited FN`: GT pixels outside the VLM boxes **and** missed by the final mask (unreachable by localization);
- `in-prompt FN`: GT inside a box but missed by SAM2;
- `FP`: predicted lesion where GT is background (over-segmentation);
- plus `recovered-outside`: GT outside the boxes that SAM2 still predicted correctly (showing boxes are not a hard ceiling).

| Arm | mIoU | loc-limited FN | in-prompt FN | FP | recovered-outside | localization share | pixel share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw SAM2 | 0.2713 | 0.4232 | 0.0922 | 0.2132 | 0.0191 | **58.1%** | 41.9% |
| FT-SAM2 | 0.5579 | 0.1374 | 0.0453 | 0.2593 | 0.2783 | **31.1%** | **68.9%** |
| FT-SAM2 + LeafClip | 0.4288 | 0.2392 | 0.1195 | 0.2124 | 0.2018 | 41.9% | 58.1% |

**Interpretation**: with zero-shot SAM2, out-of-box misses dominate the error (58%) and localization is the bottleneck. After fine-tuning, the decoder learns to extend beyond the prompt box (recovered-outside rises from 0.019 to 0.278), the localization-limited residual falls to 31%, and the remaining 69% comes from in-box misses and over-segmentation (especially FP=0.259). Leaf clipping trades FP for FN: it gives an almost unbiased area estimate but lowers mIoU by 0.129.

Representative cases (four-panel figures in `figures/qual_overlays_t4b/`, 300 dpi; green dashed lines are non-oracle VLM prompt boxes):
- Case 3 `bean_mosaic_virus_google_0098`: GT 37.4%, Arm 3 = 38.1%, IoU 0.95 (Raw 30.1%, IoU 0.36).
- Case 4 `tobacco_mosaic_virus_117`: GT 82.1%, Arm 3 = 70.0%, IoU 0.84 (Raw 17.2%, IoU 0.21); large diffuse lesions remain under-estimated.
- Case 5 `basil_downy_mildew_google_0067`: GT 19.5%, Arm 3 = 19.7%, IoU 0.85 (Raw 15.7%, IoU 0.16).

---

## 4. Delivered files

- Cache: `results/t4b_vlm_prompts_cache.json`, `results/t4b_vlm_fetch.log`.
- Per-sample: `results/t4b_e2e_preds_plantseg100.csv` (100 rows); masks `results/t4b_masks/*.npz` (100 files).
- Summary: `results/t4b_e2e_summary.csv`, `results/t4b_e2e_summary_audit.json`, `results/benchmark_v7_summary.csv` (the original 15 rows were left untouched and 4 rows were appended, giving 19 rows).
- Figures: `figures/qual_overlays_t4b/t4b_*.png` (three 300 dpi four-panel figures).
- Scripts: `scripts/fetch_t4b_vlm_prompts.py`, `scripts/eval_t4b_e2e_non_oracle.py`, `scripts/make_t4b_overlays.py`.
- The frozen manifest, `results/v7_preds_plantseg100.csv`, `results/t4_upperbound_*` and the original v4/v5/v6 files were not modified.

## 5. Limitations

1. Only PlantSeg-100 (n=100) was used; no bootstrap confidence intervals were computed, but Pearson p-values are reported.
2. The four arms share one VLM localization batch, but comparisons against v7/Pure VLM/T4 are cross-run (different TTA/calibration/prompt protocols).
3. `temperature=0` was sent to DeepSeek, but whether the provider applies it deterministically is outside our control; the published cache fixes every prompt used in this study.
4. T4 Oracle uses a single-box protocol and is not a theoretical upper bound for multi-box non-oracle results. A strict oracle upper bound would require a separate "GT multi-box + same protocol" diagnostic; this was not performed here to avoid introducing GT into the prompt pipeline.
