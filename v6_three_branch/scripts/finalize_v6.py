#!/usr/bin/env python3
"""V6 finalize: clean the resumable CSV (drop accidental repeated header rows),
compute original-scale four-way statistics + pixel metrics, and write the
canonical stats_summary_v6.csv consumed by the R plotting script.

Also emits a compact JSON blob for the report.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats

BASE = "./v6_three_branch"
P100 = BASE + "/results/predictions_v6_plantseg100.csv"
LEG = BASE + "/results/predictions_v6_legacy37.csv"
OUTSUM = BASE + "/results/stats_summary_v6.csv"
OUTJSON = BASE + "/results/v6_summary.json"


def clean(path, key="sample_id"):
    d = pd.read_csv(path)
    d = d[d[key].astype(str) != key].copy()
    d.to_csv(path, index=False)
    return d


def num(s):
    return pd.to_numeric(s, errors="coerce").values.astype(float)


def ccc(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    return 2 * np.cov(x, y)[0, 1] / (np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2)


def statrow(y, v):
    ok = np.isfinite(y) & np.isfinite(v)
    y, v = y[ok], v[ok]
    if len(y) < 3 or np.std(v) == 0:
        return dict(n=int(ok.sum()), r=float("nan"), rho=float("nan"),
                    ccc=float("nan"), mae=float(np.mean(np.abs(y - v))) if ok.sum() else float("nan"),
                    p=float("nan"))
    return dict(n=int(ok.sum()), r=float(stats.pearsonr(y, v)[0]),
                rho=float(stats.spearmanr(y, v)[0]), ccc=float(ccc(y, v)),
                mae=float(np.mean(np.abs(y - v))),
                p=float(stats.pearsonr(y, v)[1]))


rows = []
d = clean(P100)
y = num(d["published"])
methods = [("imagej", "ij_frac"), ("pure_vlm", "vlm_frac"),
           ("pure_sam2", "sam2_frac"), ("agent_v6", "agent_frac")]
p100stats = {}
for name, col in methods:
    s = statrow(y, num(d[col]))
    s["mean_iou"] = float(np.nanmean(num(d[col.replace("_frac", "_iou")])))
    s["mean_dice"] = float(np.nanmean(num(d[col.replace("_frac", "_dice")])))
    p100stats[name] = s
    rows.append(dict(domain="plantseg100_raw", method=name, **s))

# legacy37
legstats = {}
if os.path.exists(LEG):
    dl = clean(LEG)
    if "z_pred" in dl.columns:
        s = statrow(num(dl["z_pub"]), num(dl["z_pred"]))
        legstats["agent_v6"] = s
        rows.append(dict(domain="legacy37_zscore", method="agent_v6", **s))
    legstats["n_branch3"] = int((dl["branch_v6"].astype(str) == "sam2").sum()) if "branch_v6" in dl.columns else 0
    legstats["scene_all_controlled"] = bool((dl["scene"].astype(str) == "controlled").all()) if "scene" in dl.columns else None

sumdf = pd.DataFrame(rows)
sumdf.to_csv(OUTSUM, index=False)

# routing composition + prompt source
route = dict(
    scene=d["route_scene"].astype(str).value_counts().to_dict(),
    branch=d["branch"].astype(str).value_counts().to_dict(),
    prompt_source=d["prompt_source"].astype(str).value_counts().to_dict(),
    n=int(len(d)),
    mean_seconds=float(np.nanmean(num(d["seconds"]))),
)
payload = dict(plantseg100=p100stats, legacy37=legstats, routing=route)
json.dump(payload, open(OUTJSON, "w"), indent=2)

print("=== v6 FINAL stats_summary ===")
print(sumdf.to_string(index=False))
print("\n=== routing composition ===")
print(json.dumps(route, indent=2))
print("\n=== legacy37 ===")
print(json.dumps(legstats, indent=2))
print("\nWrote", OUTSUM, "and", OUTJSON)
