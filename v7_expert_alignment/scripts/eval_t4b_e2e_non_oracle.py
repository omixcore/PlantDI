#!/usr/bin/env python3
"""T4b - strictly paired, non-oracle end-to-end evaluation on PlantSeg-100.

All four arms consume the SAME frozen t4b_vlm_prompts_cache.json, produced by
one image-only primary->retry VLM pass (no TTA, no test GT used as prompt).

  Arm 1 t4b_paired_vlm_only        : VLM polygons/points rasterized + T3 leaf clip
  Arm 2 t4b_e2e_raw_sam2           : cached VLM boxes/points -> zero-shot SAM2 tiny
  Arm 3 t4b_e2e_ft_sam2            : cached VLM boxes/points -> FT SAM2 decoder
  Arm 4 t4b_e2e_ft_sam2_leafclip   : Arm 3 mask & T3 leaf_mask

Per-box masks are selected by SAM2's own predicted IoU (multimask_output=True,
argmax of returned scores); NO ground truth is consulted. Masks from multiple
boxes are unioned. Every predicted mask is saved as npz for reproducible figs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "sam2"))

from metrics import summarize  # noqa: E402
from polygon_and_clip import leaf_mask, polygons_to_mask  # noqa: E402
from eval_t4_upperbound import build_model  # noqa: E402

ARMS = ["t4b_paired_vlm_only", "t4b_e2e_raw_sam2",
        "t4b_e2e_ft_sam2", "t4b_e2e_ft_sam2_leafclip"]

PARADIGMS = {
    "t4b_paired_vlm_only": "End-to-End Deployable (Non-Oracle VLM polygon prompt; SAM-free control)",
    "t4b_e2e_raw_sam2": "End-to-End Deployable (Non-Oracle VLM prompt + zero-shot SAM2 control)",
    "t4b_e2e_ft_sam2": "End-to-End Deployable (Non-Oracle VLM Prompt + FT-SAM2)",
    "t4b_e2e_ft_sam2_leafclip": "End-to-End Deployable (Non-Oracle VLM Prompt + FT-SAM2)",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def sam2_mask(predictor, rgb, boxes, points, max_boxes=12, max_points=6):
    """Run SAM2 on VLM prompts; choose masks by SAM2 self-score only."""
    h, w = rgb.shape[:2]
    predictor.set_image(rgb)
    union = np.zeros((h, w), dtype=bool)
    for bb in (boxes or [])[:max_boxes]:
        try:
            x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        except Exception:
            continue
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if not (x2 > x1 and y2 > y1):
            continue
        masks, scores, _ = predictor.predict(
            box=np.asarray([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=True)
        union |= masks[int(np.argmax(scores))].astype(bool)
    if not union.any():
        for p in (points or [])[:max_points]:
            try:
                xy = np.asarray([[float(p[0]), float(p[1])]], dtype=np.float32)
            except Exception:
                continue
            if not (0 <= xy[0, 0] < w and 0 <= xy[0, 1] < h):
                continue
            masks, scores, _ = predictor.predict(
                point_coords=xy, point_labels=np.asarray([1], dtype=np.int64),
                multimask_output=True)
            union |= masks[int(np.argmax(scores))].astype(bool)
    return union


def metrics_for(pub, pred_frac, pred_mask, gt):
    inter = int(np.count_nonzero(pred_mask & gt))
    union = int(np.count_nonzero(pred_mask | gt))
    total = int(pred_mask.sum() + gt.sum())
    return {
        "pred": float(pred_frac),
        "iou": float(inter / union) if union else 0.0,
        "dice": float(2 * inter / total) if total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-root", default=os.path.join(ROOT, "third_party", "sam2"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "weights", "sam2_hiera_tiny.pt"))
    ap.add_argument("--ft-weights", default=os.path.join(ROOT, "weights", "sam2_tiny_decoder_ft.pt"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "t4b_vlm_prompts_cache.json"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "plantseg100", "meta100.csv"))
    ap.add_argument("--imgdir", default=os.path.join(ROOT, "data", "plantseg100", "imgs"))
    ap.add_argument("--maskdir", default=os.path.join(ROOT, "data", "plantseg100", "masks"))
    ap.add_argument("--masks-dir", default=os.path.join(ROOT, "results", "t4b_masks"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "t4b_e2e_preds_plantseg100.csv"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "results", "t4b_e2e_summary.csv"))
    ap.add_argument("--benchmark", default=os.path.join(ROOT, "results", "benchmark_v7_summary.csv"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache = json.load(open(args.cache, encoding="utf-8"))
    records = cache["records"]
    meta = {}
    with open(args.meta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["sample_id"]] = float(r["published"])
    ids = sorted(records)
    assert len(ids) == 100, "expected 100 cached records, got %d" % len(ids)
    os.makedirs(args.masks_dir, exist_ok=True)

    ft_state = torch.load(args.ft_weights, map_location="cpu", weights_only=True)
    predictors = {
        "raw": build_model(args.sam2_root, args.checkpoint, None, args.device),
        "ft": build_model(args.sam2_root, args.checkpoint, ft_state, args.device),
    }

    detail, per_arm = [], {a: {"pub": [], "pred": [], "iou": [], "dice": []} for a in ARMS}
    seconds = {"raw": 0.0, "ft": 0.0}
    t0 = time.time()
    for i, sid in enumerate(ids):
        rec = records[sid]
        bgr = cv2.imread(os.path.join(args.imgdir, sid + ".jpg"))
        gt = cv2.imread(os.path.join(args.maskdir, sid + ".png"), cv2.IMREAD_GRAYSCALE) > 0
        if bgr is None or gt.size == 0:
            print("skip unreadable", sid, flush=True)
            continue
        if gt.shape != bgr.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (bgr.shape[1], bgr.shape[0])) > 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = leaf_mask(bgr, rec.get("leaf_bbox"))

        arm1 = polygons_to_mask(gt.shape, rec.get("polygons") or [],
                                rec.get("pos_points") or []) & lm

        tr = time.perf_counter()
        raw = sam2_mask(predictors["raw"], rgb, rec.get("boxes") or [],
                        rec.get("pos_points") or [])
        raw_dt = time.perf_counter() - tr
        seconds["raw"] += raw_dt
        tf = time.perf_counter()
        ft = sam2_mask(predictors["ft"], rgb, rec.get("boxes") or [],
                       rec.get("pos_points") or [])
        ft_dt = time.perf_counter() - tf
        seconds["ft"] += ft_dt
        arm4 = ft & lm

        pub = meta[sid]
        masks = {"t4b_paired_vlm_only": arm1, "t4b_e2e_raw_sam2": raw,
                 "t4b_e2e_ft_sam2": ft, "t4b_e2e_ft_sam2_leafclip": arm4}
        np.savez_compressed(os.path.join(args.masks_dir, sid + ".npz"),
                            gt=gt, leaf=lm, **{k: v.astype(np.uint8) for k, v in masks.items()})

        row = {"sample_id": sid, "published": pub, "gt_ratio": float(gt.mean()),
               "prompt_source": rec["prompt_source"], "n_boxes": len(rec.get("boxes") or []),
               "n_polygons": len(rec.get("polygons") or []),
               "n_points": len(rec.get("pos_points") or []),
               "sam2_raw_seconds": round(raw_dt, 3), "sam2_ft_seconds": round(ft_dt, 3)}
        for arm, pm in masks.items():
            m = metrics_for(pub, float(pm.mean()), pm, gt)
            row.update({arm + "_pred": m["pred"], arm + "_iou": m["iou"],
                        arm + "_dice": m["dice"]})
            per_arm[arm]["pub"].append(pub)
            per_arm[arm]["pred"].append(m["pred"])
            per_arm[arm]["iou"].append(m["iou"])
            per_arm[arm]["dice"].append(m["dice"])
        detail.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print("eval %d/%d  %.1fs" % (i + 1, len(ids), time.time() - t0), flush=True)

    summary = []
    for arm in ARMS:
        d = per_arm[arm]
        s = summarize(d["pub"], d["pred"], d["iou"], d["dice"])
        diff = np.asarray(d["pred"]) - np.asarray(d["pub"])
        summary.append({
            "domain": "plantseg100_raw", "method": arm,
            "paradigm": PARADIGMS[arm], "status": "COMPUTED", "n": s["n"],
            "r": s["r"], "rho": s["rho"], "ccc": s["ccc"], "mae": s["mae"],
            "p": s["p"], "mean_iou": s["mean_iou"], "mean_dice": s["mean_dice"],
            "mean_bias": float(diff.mean()), "n_over": int((diff > 0).sum()),
            "seconds_per_img": round(seconds["ft"] / len(detail), 3)
            if "ft_sam2" in arm and "raw" not in arm else
            (round(seconds["raw"] / len(detail), 3) if "raw_sam2" in arm else ""),
            "input_tokens": sum(r["tokens_in"] for r in records.values()),
            "output_tokens": sum(r["tokens_out"] for r in records.values()),
            "provenance": os.path.abspath(args.cache),
            "note": "same cached non-oracle VLM prompts for all arms; no test GT used as prompt",
        })
        print("%-30s r=%.4f rho=%.4f ccc=%.4f mae=%.4f mIoU=%.4f dice=%.4f bias=%+.4f over=%d" %
              (arm, s["r"], s["rho"], s["ccc"], s["mae"], s["mean_iou"],
               s["mean_dice"], diff.mean(), int((diff > 0).sum())), flush=True)

    def write_csv(path, rows, fields):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write_csv(args.out, detail, list(detail[0].keys()))
    write_csv(args.summary, summary, list(summary[0].keys()))

    # Append to the frozen benchmark with its EXISTING header; never overwrite rows.
    with open(args.benchmark, encoding="utf-8", newline="") as fh:
        bench_fields = next(csv.reader(fh))
    with open(args.benchmark, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=bench_fields, extrasaction="ignore")
        for s in summary:
            b = dict(s)
            b["note"] = (s["note"] + "; mean_bias=%.4f; n_over=%d" %
                         (s["mean_bias"], s["n_over"]))
            w.writerow(b)

    with open(os.path.splitext(args.summary)[0] + "_audit.json", "w", encoding="utf-8") as fh:
        json.dump({
            "protocol": "T4b strictly paired non-oracle prompts",
            "n": len(detail), "elapsed_s": round(time.time() - t0, 1),
            "cache_sha256": sha256(args.cache),
            "base_ckpt_sha256": sha256(args.checkpoint),
            "ft_weights_sha256": sha256(args.ft_weights),
            "vlm_seconds_total": round(sum(r["seconds"] for r in records.values()), 1),
            "sam2_raw_seconds": round(seconds["raw"], 1),
            "sam2_ft_seconds": round(seconds["ft"], 1),
            "mask_dir": os.path.abspath(args.masks_dir),
        }, fh, indent=2)
    print("saved", args.out, args.summary, "masks", args.masks_dir, flush=True)


if __name__ == "__main__":
    main()
