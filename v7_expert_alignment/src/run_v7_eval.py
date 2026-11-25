#!/usr/bin/env python3
"""T5 - v7 dual-domain benchmark assembly + optional live evaluation.

Frozen baselines are imported from the IMMUTABLE historical CSVs (never
re-run, never edited). v7 rows are computed only from real prediction CSVs or
from a live run with a real DEEPSEEK_API_KEY and real images. Nothing is imputed.

Zero-shot (T1-T3) and supervised (T4) rows are separated by the `paradigm`
column: `Zero-Shot` vs `Supervised Domain-Adapted Upper Bound`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = os.path.dirname(HERE)
PROJECT = os.path.dirname(V7)
sys.path.insert(0, HERE)

from metrics import summarize  # noqa: E402

FIELDS = ["domain", "method", "paradigm", "n", "r", "rho", "ccc", "mae", "p",
          "mean_iou", "mean_dice", "seconds_per_img", "input_tokens",
          "output_tokens", "status", "provenance"]


def blank_row(domain: str, method: str, paradigm: str, status: str, provenance: str) -> Dict[str, object]:
    row = {f: "" for f in FIELDS}
    row.update(domain=domain, method=method, paradigm=paradigm,
               status=status, provenance=provenance)
    return row


def row_from_frozen(domain: str, method: str, paradigm: str, src: Dict[str, object],
                    provenance: str, status: str = "FROZEN_BASELINE") -> Dict[str, object]:
    row = blank_row(domain, method, paradigm, status, provenance)
    for k in ("n", "r", "rho", "ccc", "mae", "p", "mean_iou", "mean_dice"):
        if k in src and src[k] not in ("", None):
            row[k] = src[k]
    return row


def row_from_predictions(domain: str, method: str, paradigm: str, csv_path: str,
                         status: str = "COMPUTED") -> Dict[str, object]:
    rows = read_csv_rows(csv_path)
    rows = [r for r in rows if str(r.get("sample_id")) != "sample_id"]
    seen = set()
    uniq = []
    for r in rows:
        sid = str(r.get("sample_id"))
        if sid and sid not in seen:
            seen.add(sid)
            uniq.append(r)
    def col(name):
        out = []
        for r in uniq:
            try:
                out.append(float(r[name]))
            except Exception:
                pass
        return out
    y = col("published"); p = col("pred")
    s = summarize(y, p, col("iou") or None, col("dice") or None)
    row = blank_row(domain, method, paradigm, status, os.path.abspath(csv_path))
    for k, v in s.items():
        row[k] = v
    for key, dst in (("seconds", "seconds_per_img"),
                     ("tokens_in", "input_tokens"), ("tokens_out", "output_tokens")):
        vals = col(key)
        if vals:
            row[dst] = float(np.mean(vals) if dst == "seconds_per_img" else np.sum(vals))
    return row


def read_csv_rows(path: str) -> List[Dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_baselines() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    p6 = os.path.join(PROJECT, "v6_three_branch/results/stats_summary_v6.csv")
    ps = os.path.join(PROJECT, "v6_three_branch/ablation_samfree/stats_summary_samfree.csv")
    if os.path.exists(p6):
        for r in read_csv_rows(p6):
            if r["domain"] == "plantseg100_raw":
                rows.append(row_from_frozen("plantseg100_raw", r["method"], "Zero-Shot (frozen v6)",
                                            r, p6))
    if os.path.exists(ps):
        for r in read_csv_rows(ps):
            dom = r["domain"]
            rows.append(row_from_frozen(dom, r["method"], "Post-hoc SAM-free ablation",
                                        r, ps))
    return rows


# ---------------------------------------------------------------------------
# Live T1-T3 run (only when a key and real images are available)
# ---------------------------------------------------------------------------
def run_live_plantseg(meta_csv: str, imgdir: str, maskdir: str, out_csv: str,
                      model: str = "deepseek-flash",
                      calibration_csv: str = "", limit: int = 0) -> None:
    import cv2
    import pandas as pd
    from polygon_and_clip import (apply_calibration, fit_calibration,
                                  locate_and_clip)
    from metrics import mask_metrics
    from llm_client_deepseek import key_fingerprint
    from router_brain_v7 import query as route_query

    meta = pd.read_csv(meta_csv)
    if limit:
        meta = meta.head(limit)
    eval_ids = set(meta.sample_id.astype(str))

    calibration = None
    if calibration_csv:
        cal_df = pd.read_csv(calibration_csv)
        rows = cal_df[["sample_id", "symptom", "pred_fraction", "gt_fraction"]].to_dict("records")
        calibration = fit_calibration(rows, eval_ids)  # hard leakage guard

    done = set()
    if os.path.exists(out_csv):
        d0 = pd.read_csv(out_csv)
        d0 = d0[d0.sample_id.astype(str) != "sample_id"].drop_duplicates("sample_id", keep="first")
        done = set(d0.sample_id.astype(str))
        d0.to_csv(out_csv, index=False)
    header = not os.path.exists(out_csv) or not done

    print("DeepSeek key fingerprint:", key_fingerprint(), flush=True)
    for i, r in enumerate(meta.itertuples(index=False), 1):
        sid = str(r.sample_id)
        if sid in done:
            continue
        ip = os.path.join(imgdir, sid + ".jpg")
        mp = os.path.join(maskdir, sid + ".png")
        if not (os.path.exists(ip) and os.path.exists(mp)):
            print("[MISS]", sid, flush=True)
            continue
        dec, rst = route_query(ip, model=model)
        pred = locate_and_clip(ip, model=model, reasoning="high")
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        iou, dice = mask_metrics(pred["mask"], gt)
        di = apply_calibration(pred["di"], pred["symptoms"], calibration)
        row = dict(sample_id=sid, published=float(gt.sum()) / float(gt.size), pred=di,
                   iou=iou, dice=dice, branch=dec["branch"],
                   route_scene=dec["scene_type"], route_bg=dec["background"],
                   prompt_source=pred["prompt_source"],
                   tokens_in=rst.input_tokens + pred["tokens_in"],
                   tokens_out=rst.output_tokens + pred["tokens_out"],
                   seconds=round(rst.latency_s + pred["seconds"], 3))
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=header, index=False)
        header = False
        print("[%d/%d] %s pred=%.3f gt=%.3f %.1fs" %
              (i, len(meta), sid[:32], di, row["published"], row["seconds"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(V7, "results/benchmark_v7_summary.csv"))
    ap.add_argument("--legacy-v7-csv", default="", help="per-sample v7 Legacy-37 predictions")
    ap.add_argument("--plantseg-v7-csv", default="", help="per-sample v7 PlantSeg-100 predictions")
    ap.add_argument("--supervised-csv", default="", help="per-sample T4 fine-tuned predictions")
    ap.add_argument("--live", action="store_true", help="run DeepSeek live (needs key + images)")
    ap.add_argument("--meta", default="")
    ap.add_argument("--imgdir", default="")
    ap.add_argument("--maskdir", default="")
    ap.add_argument("--live-out", default="")
    ap.add_argument("--calibration-csv", default="")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Freeze audit gate: abort if any pinned artefact changed.
    sys.path.insert(0, os.path.join(V7, "scripts"))
    from verify_manifest import main as verify  # noqa: E402
    if verify() != 0:
        raise SystemExit("ABORT: frozen manifest verification failed")

    if args.live:
        if not (args.meta and args.imgdir and args.maskdir and args.live_out):
            raise SystemExit("--live requires --meta --imgdir --maskdir --live-out")
        run_live_plantseg(args.meta, args.imgdir, args.maskdir, args.live_out,
                          args.model, args.calibration_csv, args.limit)
        return

    rows = build_baselines()

    # Legacy-37: v7 policy keeps the exact v4 path for all controlled panels;
    # reuse the immutable v4 prediction table (no re-run, no fabrication).
    v4 = os.path.join(PROJECT, "results/predictions_v4_agent.csv")
    if os.path.exists(v4):
        d = read_csv_rows(v4)
        z_pub = [float(r["z_pub"]) for r in d if r.get("z_pub") not in ("", None)]
        z_pred = [float(r["z_pred"]) for r in d if r.get("z_pred") not in ("", None)]
        s = summarize(z_pub[:len(z_pred)], z_pred[:len(z_pub)])
        rows.append(row_from_frozen("legacy37_zscore", "v4_frozen_controlled_path",
                                    "Zero-Shot (policy-equivalent frozen)", s, v4,
                                    "POLICY_EQUIVALENT_FROZEN"))
        rows.append(blank_row("legacy37_zscore", "v7_zero_shot", "Zero-Shot (T1-T3)",
                              "NOT_RUN", v4 + " (reused by frozen-policy equivalence)"))
    else:
        rows.append(blank_row("legacy37_zscore", "v7_zero_shot", "Zero-Shot (T1-T3)",
                              "NOT_RUN", "v4 frozen table missing"))

    if args.legacy_v7_csv:
        rows = [r for r in rows if not (r["domain"] == "legacy37_zscore" and r["method"] == "v7_zero_shot")]
        rows.append(row_from_predictions("legacy37_zscore", "v7_zero_shot", "Zero-Shot (T1-T3)",
                                         args.legacy_v7_csv))

    rows.append(blank_row("plantseg100_raw", "v7_zero_shot", "Zero-Shot (T1-T3)",
                          "NOT_RUN", "requires DEEPSEEK_API_KEY + PlantSeg-100 images/masks"))
    if args.plantseg_v7_csv:
        rows = [r for r in rows if not (r["domain"] == "plantseg100_raw" and r["method"] == "v7_zero_shot")]
        rows.append(row_from_predictions("plantseg100_raw", "v7_zero_shot", "Zero-Shot (T1-T3)",
                                         args.plantseg_v7_csv))
    rows.append(blank_row("plantseg100_raw", "supervised_ft_sam2", "Supervised Domain-Adapted Upper Bound",
                          "NOT_RUN", "T4 requires official Train split + sampled VLM prompt cache"))
    if args.supervised_csv:
        rows = [r for r in rows if r["method"] != "supervised_ft_sam2"]
        rows.append(row_from_predictions("plantseg100_raw", "supervised_ft_sam2",
                                         "Supervised Domain-Adapted Upper Bound", args.supervised_csv))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out, "rows:", len(rows))
    for r in rows:
        print("  %-18s %-28s %-40s r=%-10s status=%s" %
              (r["domain"], r["method"], r["paradigm"], r["r"] or "-", r["status"]))


if __name__ == "__main__":
    main()
