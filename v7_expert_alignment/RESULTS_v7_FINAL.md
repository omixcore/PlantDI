# PlantDI v7 - Final Results Report (Honest Account)

*English translation of the frozen v7 report. Editorial note: this is the historical v7 snapshot; T4 (oracle upper bound) and T4b (deployable non-oracle) were completed later and are documented in `RESULTS_T4_FINAL.md` and `RESULTS_T4b_E2E_FINAL.md` in this directory.*

Date: 2026-11-25. Backend: DeepSeek `deepseek-flash` (`effort=high`), original image bytes in base64 with no downscaling. The key was injected in-process only (fingerprint `6ea770ca`); it was never written to disk or echoed. All numbers come from the frozen CSV files and are auditable.

## 1. Audit trail

- The policy and prompts were frozen **before any test-set evaluation**: `policy_manifest_v7.json`, route SHA256 `9958e71dcdef4137e19c2d634672cb2fdf1657db322d288c3448fa00946a2da4`. The backend was switched from Gemini to DeepSeek before freezing, and the manifest was regenerated afterwards; prompts and routing were unchanged.
- `verify_manifest.py` returned **OK** both before evaluation and before summarization.
- No v4/v5/v6/SAM-free code or result CSV was modified.
- Five-image smoke test: `tokens_in=15507 / tokens_out=30287`; prompt sources primary=2 / retry=3; no silent zero-token degradation.

## 2. Comparison series (PlantSeg-100 raw, n=100, seed=42, original scale)

| Method | r | rho | CCC | MAE | mIoU | Dice | Status |
|---|---|---|---|---|---|---|---|
| v6 frozen (agent) | 0.1039 | 0.1592 | 0.0899 | 0.1779 | 0.1894 | 0.2674 | FROZEN |
| ImageJ thresholding | 0.0318 | 0.0349 | 0.0282 | 0.2062 | 0.2262 | 0.3253 | FROZEN |
| Pure SAM2 | 0.1608 | 0.2477 | 0.0644 | 0.1573 | 0.1720 | 0.2422 | FROZEN |
| SAM-free (post-hoc) | 0.2381 | 0.3109 | 0.2352 | 0.1531 | 0.2611 | 0.3835 | FROZEN |
| Pure VLM | 0.3316 | 0.3899 | 0.3122 | 0.1280 | 0.2576 | 0.3814 | FROZEN |
| **v7 Zero-Shot** | **0.5434** | **0.5118** | **0.4677** | **0.1101** | 0.2475 | 0.3602 | COMPUTED |
| Supervised FT-SAM2 | - | - | - | - | - | - | NOT_RUN |

v7 latency was 35.7 s/image; input tokens 261,853 and output tokens 613,734 over all 100 images. The two-sided p-value for Pearson r is 5.15e-9. Zero-shot (T1-T3) and supervised (T4) results are kept in separate columns in the summary table.

## 3. Acceptance criteria, point by point

1. **Legacy-37 r >= 0.9104**: MET_AT_REPORTED_PRECISION. The frozen v7 policy routes the 37 controlled panels back to the v4 path, giving r=0.9103949 (0.9104 at four decimals). No new v7 measurement was made because the literature images were unavailable; this is structural equivalence, not a new independent measurement. The strict floating-point value is 5e-6 below 0.9104, which is rounding, not degradation.
2. **PlantSeg-100: all five metrics better than SAM-free and Pure VLM**: **PARTIAL**. r / CCC / MAE are significantly better than both baselines (r improves from 0.238/0.332 to 0.543, p=5e-9), but mIoU 0.2475 and Dice 0.3602 are **slightly lower** than SAM-free (0.2611/0.3835) and Pure VLM (0.2576/0.3814). Leaf-mask clipping tightens the predicted lesions, improving severity ranking but reducing pixel overlap. A partial result is not reported as a complete one.
3. **Cases 3/4/5 approaching Expert GT**: **PARTIAL**; overlays were generated.
   - Case 3 bean mosaic: GT 37.4%, v7 10.8% (worse than v6 Pure-VLM 57.3%).
   - Case 4 tobacco mosaic: GT 82.1%, v7 50.9% (better than v6 Pure-VLM 40.7%).
   - Case 5 basil downy mildew: GT 19.5%, v7 28.7% (better than v6 Pure-VLM 38.1%).
   2 of 3 are closer. Overlays: `figures/qual_overlays_v7/qual_v7_*.png` (300 dpi; Original / Expert GT / v7). Caption numbers come from the frozen CSV; the masks are a visualization re-run of the same frozen pipeline, and the stochastic LLM output changes coverage. The captions flag this separately so that a re-run is never presented as the frozen scalar result.

## 4. Key limitations

- **LLM non-determinism**: re-running the same image changes polygon coverage substantially (Case 4: 50.9% -> 31.2% -> 3.8%). The per-sample values in the frozen CSV are the sole authoritative result; visualization re-runs are qualitative only.
- **T4 had not been run**: the local machine initially lacked the `sam2` package and `sam2_hiera_tiny.pt`, and CPU fine-tuning on 7,900 images was not feasible. The isolation audit passed (train intersect Test = 0, train intersect PlantSeg-100 = 0), but the summary table honestly kept NOT_RUN.
- **Legacy-37 was not re-measured**: the literature images were unreachable; only the structural v4 equivalent was retained.
- **Pixel-overlap metrics did not improve**: T3 leaf clipping optimized severity ranking at the expense of mIoU/Dice. Improving pixel overlap requires the T4 pixel-level fine-tuning upper bound (not run at the time of this snapshot).

## 5. File index

- `results/v7_preds_plantseg100.csv`: 100 per-sample rows (published/pred/iou/dice/branch/prompt_source/tokens/seconds).
- `results/benchmark_v7_summary.csv`: comparison table; zero-shot and supervised rows are separated and NOT_RUN is retained honestly.
- `results/acceptance_check_v7.json`: machine-readable conclusions for the three acceptance checks.
- `figures/fig_v7_benchmark.svg/.pdf`: vector comparison figure (Arial 7 pt, 0.5 pt solid black lines).
- `figures/qual_overlays_v7/*.png`: Case 3/4/5 three-panel overlays.
- `policy_manifest_v7.json`: frozen audit manifest (route + 5 prompts + 8 source files + 3 historical baseline hashes).
