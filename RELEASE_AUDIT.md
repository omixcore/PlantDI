# PlantDI GitHub Release - Pre-Push Audit

Date: 2026-11-25. Target repository: `omixcore/PlantDI`. Staging directory: `PlantDI_github_release/` (sibling of the original `PlantDI_MVP/`).

No file in the original working directory was modified. All copies in this staging tree were made from the frozen originals; documentation in this tree is entirely in English.

## 1. Whitelist and policy hashes

Copied: v4/v6/v7/T4/T4b Python sources, `tests/offline_selftest.py`, `src/make_figures.R`, frozen manifests/audits, historical baseline CSVs, frozen v7/T4/T4b result tables and caches, and the three English result reports. `scripts/launch_with_key.py` was excluded; every retained script reads the key only from the standard `DEEPSEEK_API_KEY` environment variable.

| Frozen object | SHA256 | Status |
|---|---|---|
| v4 route policy | `a103bd6208334ba0c6b8bd8f25916fdaaa028c8e8e0a44f057712cb76b4fa4df` | verified by running `tmp_qc/frozen_policy.py` |
| v6 route policy | `be104b440f7151eeba4e6fa8034967d80492760a521fa9bd58526cdf810a084b` | verified by running `frozen_policy_v6.py` |
| v7 route (manifest) | `9958e71dcdef4137e19c2d634672cb2fdf1657db322d288c3448fa00946a2da4` | verified by `frozen_policy_v7.py` |
| T4b VLM prompt cache | `782fcc602677bacca9aa24530c57af3906baddbba831ad5c51091a09778c39c2` | verified by file SHA256 |

## 2. Release assets (outside the Git tree)

| Asset | SHA256 | Required value |
|---|---|---|
| `sam2_tiny_decoder_ft.pt` | `7a7879fa4ffd235a63c550201d9ce539b20bdba4df01bbd6b26a91c8c5aaa370` | matches exactly |
| `t4b_masks_plantseg100.zip` | `997136763bc4d9841a6e86b4d3a826eb7e336a286325df6eb27464a5da26e24b` | 100 npz files |

## 3. Mandatory pre-push checks

### Check 1 - Secret leakage: PASS

Regular-expression scan of the whole staging tree:

| Pattern class (literal tokens omitted to keep the scan at zero) | Matching files |
|---|---:|
| DeepSeek-style key prefix | 0 |
| Google API key prefix | 0 |
| Legacy Gemini key prefix | 0 |
| DeepSeek key assignment in source | 0 |
| Gemini key assignment in source | 0 |
| Bearer-token header with a literal token | 0 |
| GitHub token prefixes | 0 |

Only the non-sensitive API-key fingerprint `6ea770ca` remains, inside the frozen VLM prompt cache.

### Check 2 - Large files and private paths: PASS

- Files larger than 25 MB in the Git tree: **0**.
- Local Windows drive paths, the former Linux server prefix, internal private-network addresses and user-home paths: **0 after sanitization** (literal examples intentionally omitted so the scan stays at zero).
- Sanitization affected 23 v4/v6 script copies (private server prefix replaced by a relative path prefix) and five result/audit JSON/CSV copies (absolute provenance paths replaced by repository-relative paths). The three baselines pinned by `policy_manifest_v7.json` were not touched.

### Check 3 - Frozen manifest and offline self-test: PASS

```text
$ python v7_expert_alignment/scripts/verify_manifest.py
MANIFEST VERIFICATION: OK
route sha256: 9958e71dcdef4137e19c2d634672cb2fdf1657db322d288c3448fa00946a2da4

$ python v7_expert_alignment/tests/offline_selftest.py
MANIFEST VERIFICATION: OK
... "policy_sha256": "9958e71d...", "manifest_verification": "OK"
```

### Check 4 - File tree and size report: PASS

- Tracked files: **86**; total size: **871,783 bytes** (about 0.83 MB).
- Top 10 largest files:

| Rank | Bytes | Path |
|---:|---:|---|
| 1 | 339,692 | `v7_expert_alignment/results/t4b_vlm_prompts_cache.json` |
| 2 | 45,624 | `v6_three_branch/results/predictions_v6_plantseg100.csv` |
| 3 | 37,943 | `v7_expert_alignment/results/t4_upperbound_detail.json` |
| 4 | 33,307 | `v7_expert_alignment/results/t4b_e2e_preds_plantseg100.csv` |
| 5 | 17,757 | `v7_expert_alignment/results/v7_preds_plantseg100.csv` |
| 6 | 17,704 | `v4_controlled/results/trackC_predictions.csv` |
| 7 | 13,382 | `v7_expert_alignment/src/polygon_and_clip.py` |
| 8 | 12,124 | `v4_controlled/scripts/01_mllm_sam2_pipeline.py` |
| 9 | 11,387 | `v7_expert_alignment/scripts/eval_t4b_e2e_non_oracle.py` |
| 10 | 11,212 | `v4_controlled/scripts/06_final_all_images.py` |

Top-level tree:

```text
.
├── README.md, LICENSE, requirements.txt, .gitignore, .gitattributes, RELEASE_AUDIT.md
├── results/predictions_v4_agent.csv
├── v4_controlled/            frozen v4 policy, results, scripts
├── v5_formal/results/        pinned baseline
├── v6_three_branch/          scripts, results, SAM-free ablation
└── v7_expert_alignment/      manifest, English reports, data audits,
                              results/caches, scripts, src, tests
```

## 4. Git commands

The staging tree is already initialized and committed locally (`c32a6e6`). To publish:

```bash
cd <path>/PlantDI_github_release
git remote add origin https://github.com/omixcore/PlantDI.git
git push -u origin master
```

If the repository does not yet exist, create it as a public repository named `PlantDI` under the `omixcore` account before running the push.

**Current blocker:** a remote probe returned `Repository not found` for `omixcore/PlantDI`, and the environment's network security policy blocked further remote access. The local commit is complete; only the final push remains.
