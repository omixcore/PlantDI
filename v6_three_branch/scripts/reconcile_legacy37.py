#!/usr/bin/env python3
"""Reconcile: propagate the honest Branch3 rescore into the canonical legacy37 CSV.

fix_legacy37_branch3.py produced predictions_v6_legacy37_true.csv (real SAM2 pred
for the 7 Branch3 images) and an audit CSV. This script makes the whole chain
self-consistent:
  1. back up the old reuse-based CSV  -> predictions_v6_legacy37_reuse_version.csv
  2. overwrite the canonical predictions_v6_legacy37.csv with the true preds,
     recomputing z_pub / z_pred within batch (same transform as v4)
  3. so finalize_v6.py and stats_plots_v6.R (which read the canonical CSV)
     will now report the honest r=0.4590 instead of the stale 0.9104.
"""
import os, json
import numpy as np, pandas as pd

BASE = "./v6_three_branch"
LEG = BASE + "/results/predictions_v6_legacy37.csv"
BAK = BASE + "/results/predictions_v6_legacy37_reuse_version.csv"
TRUE = BASE + "/results/predictions_v6_legacy37_true.csv"

# back up the version that (incorrectly) reused v4 color for Branch3
pd.read_csv(LEG).to_csv(BAK, index=False)

d = pd.read_csv(TRUE)
d = d[d.sample_id.astype(str) != "sample_id"].reset_index(drop=True)
# keep canonical columns only (drop the fix-script helper columns pred_old/pred_source)
keep = [c for c in d.columns if c not in ("pred_old", "pred_source")]
d = d[keep].copy()
# recompute z within batch, exactly as v4/backtest
d["z_pub"] = d.groupby("batch").published.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
d["z_pred"] = d.groupby("batch").pred.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
d.to_csv(LEG, index=False)
print("reconciled canonical CSV:", LEG, "rows=", len(d))
print("cols:", list(d.columns))
print("z-r recomputed will be produced by finalize/R.")

# re-attach the rescore comparison metadata into summary (finalize will have run before this in caller order)
