#!/usr/bin/env python3
"""Re-attach Branch3 honest-rescore audit into v6_summary.json (finalize may overwrite it)."""
import json, pandas as pd
BASE = "./v6_three_branch"
sp = BASE + "/results/v6_summary.json"
ad = pd.read_csv(BASE + "/results/legacy37_branch3_sam2_audit.csv")
summ = json.load(open(sp))
summ["legacy37"]["branch3_rescore"] = {
    "old_r": 0.9103949082059803,
    "new_r": 0.4590465228975341,
    "delta_r": -0.4513483853084462,
    "old_rho": 0.8871235724885476,
    "new_rho": 0.406924365247353,
    "old_ccc": 0.9103949082059805,
    "new_ccc": 0.45904652289753406,
    "old_mae": 0.34614963509672525,
    "new_mae": 0.7663595698924228,
    "reason": "backtest originally reused v4 color pred for all 37; the 7 images routed to Branch3(SAM2) were re-scored with REAL SAM2 inference",
    "numerator": "SAM2 mask lesion pixels (real inference)",
    "denominator": "v4 leaf_px (same denominator as other 30 controlled images)",
    "n_branch3": len(ad),
    "prompt_sources": ad.prompt_source.value_counts().to_dict(),
    "per_image": ad[["sample_id", "published", "v4_color_pred", "sam2_di_pct", "prompt_source"]].to_dict("records"),
}
json.dump(summ, open(sp, "w"), indent=2)
print("patched branch3_rescore into summary; audit rows =", len(ad))
