#!/usr/bin/env python3
"""Fair subset analysis: separate availability bias from fallback quality.

v5 SAM2 r=0.540 was computed on only 18/48 images that HAD a VLM lesion box
(availability-biased subset). v6 covers 100/100 via fallback. To compare fairly
we score v6 on: (a) v5's 48-image subset, (b) all 100, and report prompt_source.
"""
import numpy as np, pandas as pd
from scipy import stats

BASE = "./v6_three_branch"
V5 = "./v5_formal/results/trackC_predictions.csv"
P = BASE + "/results/predictions_v6_plantseg100.csv"


def num(s):
    return pd.to_numeric(s, errors="coerce").values.astype(float)


def stat(y, v, tag):
    ok = np.isfinite(y) & np.isfinite(v)
    if ok.sum() < 3 or np.std(v[ok]) == 0:
        print("%-28s n=%d (degenerate)" % (tag, ok.sum())); return
    r = stats.pearsonr(y[ok], v[ok])[0]
    rho = stats.spearmanr(y[ok], v[ok])[0]
    print("%-28s n=%3d r=%+.3f rho=%+.3f" % (tag, ok.sum(), r, rho))


d = pd.read_csv(P)
d = d[d.sample_id.astype(str) != "sample_id"].copy()
d["y"] = num(d.published)
d["ag"] = num(d.agent_frac)
d["sm"] = num(d.sam2_frac)
d["ij"] = num(d.ij_frac)

# which v5 images were SAM2-valid (had VLM lesions) vs empty
v5 = pd.read_csv(V5)
v5_valid = set(v5.loc[v5.sam2_frac.notna(), "sample_id"])
v5_fail = set(v5.loc[v5.sam2_frac.isna(), "sample_id"])
d["v5_had_lesion"] = d.sample_id.isin(v5_valid).map({True: "v5_valid18", False: "other"})
d.loc[d.sample_id.isin(v5_fail), "v5_had_lesion"] = "v5_failed30"
d.loc[~d.sample_id.isin(v5_valid | v5_fail), "v5_had_lesion"] = "v6_new52"

print("=== v6 agent r by subset ===")
stat(d.y.values, d.ag.values, "ALL 100")
sub = d[d.in_v5_48.astype(str).isin(["1", "1.0", "True"])]
stat(sub.y.values, sub.ag.values, "v5-48 subset")
for g, gg in d.groupby("v5_had_lesion"):
    stat(gg.y.values, gg.ag.values, g)

print("\n=== v5 reference (published report) ===")
print("v5 SAM2 (n=18 valid) r=+0.540 ; v5 agent (n=48) r=-0.118")
print("\n=== prompt_source x subset ===")
print(pd.crosstab(d.v5_had_lesion, d.prompt_source).to_string())

print("\n=== per-disease worst/best (v6 agent abs error) ===")
d["ae"] = (d.y - d.ag).abs()
g = d.groupby("disease").agg(n=("ae", "size"), mae=("ae", "mean"), gt=("y", "mean")).sort_values("mae", ascending=False)
print("worst 8:"); print(g.head(8).round(3).to_string())
print("best 5:"); print(g.tail(5).round(3).to_string())
