#!/usr/bin/env python3
"""FIX (methodology): legacy37 Branch3 samples must be ACTUALLY segmented by SAM2.

Problem: backtest_legacy37.py reused v4 color pred for every row, including the 7
controlled images that the frozen v6 policy routed to Branch3(SAM2). That is a
methodological mismatch: nominal route = SAM2, scored value = v4 color. A reviewer
would reject it. This script truly runs the v6 Branch3 (SAM2 + two-level prompt
fallback) on those 7 images, and rescores legacy37 honestly.

DI definition kept IDENTICAL to v4 for comparability:
    DI% = 100 * lesion_px / leaf_px
  - numerator lesion_px = pixels of the REAL SAM2 mask (this fix)
  - denominator leaf_px = v4's leaf pixel count from predictions_final.csv
    (same denominator used by all 30 other controlled images -> batch z-score
     stays comparable; only the numerator method changes for these 7)
Nothing is fabricated; negative degradation is reported as-is.

Writes:
  results/legacy37_branch3_sam2_audit.csv  (per-image audit)
  results/predictions_v6_legacy37_true.csv  (full 37, pred replaced for the 7)
  updates results/v6_summary.json [legacy37] and stats_summary_v6.csv legacy row
"""
import os, sys, json
import numpy as np, pandas as pd, cv2
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from sam2_fallback import locate_with_fallback, mask_from_boxes_points
from frozen_policy_v6 import policy_hash

ROOT = "."
BASE = ROOT + "/v6_three_branch"
IMG = ROOT + "/data/continuous_leaves"
PRED = BASE + "/results/predictions_v6_legacy37.csv"
FINAL = ROOT + "/results/predictions_final.csv"
AUDIT = BASE + "/results/legacy37_branch3_sam2_audit.csv"
OUT = BASE + "/results/predictions_v6_legacy37_true.csv"
CKPT = ROOT + "/models/sam2_hiera_tiny.pt"
key = os.environ["DEEPSEEK_API_KEY"]


def ccc(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]; y = np.asarray(y)[ok]
    return 2 * np.cov(x, y)[0, 1] / (np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2)


def score(d, predcol="pred"):
    z_pub = d.groupby("batch").published.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    z_pred = d.groupby("batch")[predcol].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    r, p = stats.pearsonr(z_pub, z_pred)
    rho, _ = stats.spearmanr(z_pub, z_pred)
    cc = ccc(z_pub.values, z_pred.values)
    mae = float((z_pub - z_pred).abs().mean())
    return dict(n=len(d), r=float(r), rho=float(rho), ccc=float(cc), mae=float(mae), p=float(p))


def main():
    print("FROZEN POLICY SHA256:", policy_hash(), flush=True)
    d = pd.read_csv(PRED)
    # clean any repeated-header residue defensively
    d = d[d.sample_id.astype(str) != "sample_id"].reset_index(drop=True)
    s7 = d[d.branch_v6 == "sam2"].copy()
    print("Branch3(SAM2) controlled images to truly segment:", len(s7), flush=True)

    final = pd.read_csv(FINAL)
    leaf = dict(zip(final.sample_id, final.leaf_px))
    v4pred = dict(zip(final.sample_id, final.pred))

    # load SAM2 once
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", CKPT, device="cpu"))
    print("SAM2 loaded", flush=True)

    audit = []
    true_pred = {}
    for i, row in s7.iterrows():
        sid = row.sample_id
        p = IMG + "/" + sid + ".png"
        bgr = cv2.imread(p)
        if bgr is None:
            print("!! image unreadable", sid, flush=True)
            continue
        boxes, points, psrc, tok, scale, ann = locate_with_fallback(p, key)
        sm = mask_from_boxes_points(sam, bgr, boxes, points)  # bool mask of lesion
        sam_lesion_px = int(sm.sum())
        lpx = int(leaf.get(sid, 0))
        di = 100.0 * sam_lesion_px / max(lpx, 1)
        true_pred[sid] = di
        rec = dict(sample_id=sid, batch=row.batch, published=float(row.published),
                   v4_color_pred=float(row.pred), v4_leaf_px=lpx,
                   sam2_lesion_px=sam_lesion_px, sam2_di_pct=di,
                   prompt_source=psrc, n_boxes=len(boxes), n_points=len(points),
                   tokens=tok)
        audit.append(rec)
        print("[%d/%d] %-32s src=%-8s boxes=%d pts=%d sam_px=%d/%d DI=%.2f%% (v4 color=%.2f%%)" % (
            len(audit), len(s7), sid[:32], psrc, len(boxes), len(points),
            sam_lesion_px, lpx, di, float(row.pred)), flush=True)

    ad = pd.DataFrame(audit)
    ad.to_csv(AUDIT, index=False)

    # build corrected full dataframe
    d["pred_old"] = d["pred"]
    d["pred_source"] = np.where(d.sample_id.isin(true_pred), "sam2_true", "v4_reuse")
    d["pred"] = d.apply(lambda r: true_pred.get(r.sample_id, r.pred), axis=1)
    d.to_csv(OUT, index=False)

    old = score(d, "pred_old")
    new = score(d, "pred")
    print("\n=== legacy37 honest rescoring ===")
    print("OLD (reuse v4 color for Branch3):", old)
    print("NEW (real SAM2 for Branch3):     ", new)
    print("delta r = %+.4f" % (new["r"] - old["r"]), flush=True)
    print("audit:\n", ad.to_string(index=False), flush=True)

    # update summary json
    sp = BASE + "/results/v6_summary.json"
    summ = json.load(open(sp))
    summ["legacy37"]["agent_v6"] = new
    summ["legacy37"]["branch3_rescore"] = dict(
        old=old, new=new, method="real SAM2 Branch3 for 7 controlled images",
        numerator="SAM2 mask lesion pixels", denominator="v4 leaf_px (same as other 30)",
        prompt_sources=ad.prompt_source.value_counts().to_dict())
    json.dump(summ, open(sp, "w"), indent=2)

    # update stats_summary_v6.csv legacy row
    st = pd.read_csv(BASE + "/results/stats_summary_v6.csv")
    mask = (st.domain == "legacy37_zscore") & (st.method == "agent_v6")
    st.loc[mask, ["r", "rho", "ccc", "mae", "p"]] = [new["r"], new["rho"], new["ccc"], new["mae"], new["p"]]
    st.to_csv(BASE + "/results/stats_summary_v6.csv", index=False)
    print("stats files updated.", flush=True)


if __name__ == "__main__":
    main()
