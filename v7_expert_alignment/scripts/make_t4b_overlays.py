#!/usr/bin/env python3
"""T4b qualitative overlays: four columns, strictly from the saved npz masks.

Original | Expert GT | VLM Prompt + Raw SAM2 | VLM Prompt + FT-SAM2
Green thin dashed boxes are the cached NON-ORACLE VLM prompt boxes (full res).
Caption numbers are read from the frozen t4b_e2e_preds CSV, never re-inferred.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "data", "plantseg100", "imgs")
MSKS = os.path.join(ROOT, "results", "t4b_masks")
CACHE = os.path.join(ROOT, "results", "t4b_vlm_prompts_cache.json")
PREDS = os.path.join(ROOT, "results", "t4b_e2e_preds_plantseg100.csv")
OUTD = os.path.join(ROOT, "figures", "qual_overlays_t4b")
CASES = ["plantseg_bean_mosaic_virus_google_0098",
         "plantseg_tobacco_mosaic_virus_117",
         "plantseg_basil_downy_mildew_google_0067"]

plt.rcParams.update({"font.family": "Arial", "font.size": 7,
                     "svg.fonttype": "none", "text.color": "black"})


def overlay(bgr, mask, color):
    out = bgr.copy()
    m = mask.astype(bool)
    layer = np.zeros_like(bgr)
    layer[:] = color
    out[m] = (0.45 * out[m] + 0.55 * layer[m]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def dashed_rect(bgr, box, color=(0, 255, 0), thickness=1, dash=7, gap=4):
    h, w = bgr.shape[:2]
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    except Exception:
        return
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay))
        step = dash + gap
        for s in range(0, length, step):
            e = min(s + dash, length)
            if abs(bx - ax) >= abs(by - ay):
                p1 = (ax + s, ay); p2 = (ax + e, ay)
            else:
                p1 = (ax, ay + s); p2 = (ax, ay + e)
            cv2.line(bgr, p1, p2, color, thickness, cv2.LINE_AA)


def main():
    os.makedirs(OUTD, exist_ok=True)
    cache = json.load(open(CACHE, encoding="utf-8"))["records"]
    rows = {r["sample_id"]: r for r in csv.DictReader(open(PREDS, encoding="utf-8"))}
    for sid in CASES:
        bgr = cv2.imread(os.path.join(IMG, sid + ".jpg"))
        z = np.load(os.path.join(MSKS, sid + ".npz"))
        gt = z["gt"].astype(bool)
        raw = z["t4b_e2e_raw_sam2"].astype(bool)
        ft = z["t4b_e2e_ft_sam2"].astype(bool)
        r = rows[sid]

        raw_v = overlay(bgr, raw, (255, 120, 0))
        ft_v = overlay(bgr, ft, (255, 120, 0))
        for bb in cache[sid]["boxes"]:
            dashed_rect(raw_v, bb)
            dashed_rect(ft_v, bb)

        panels = [
            (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "Original"),
            (overlay(bgr, gt, (0, 0, 255)), "Expert GT\nDI=%.1f%%" % (100 * float(r["published"]))),
            (raw_v, "VLM Prompt + Raw SAM2\nDI=%.1f%%  IoU=%.2f" %
             (100 * float(r["t4b_e2e_raw_sam2_pred"]), float(r["t4b_e2e_raw_sam2_iou"]))),
            (ft_v, "VLM Prompt + FT-SAM2\nDI=%.1f%%  IoU=%.2f" %
             (100 * float(r["t4b_e2e_ft_sam2_pred"]), float(r["t4b_e2e_ft_sam2_iou"]))),
        ]
        fig, axs = plt.subplots(1, 4, figsize=(13.6, 3.3))
        for ax, (img, title) in zip(axs, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
        fig.suptitle(sid.replace("plantseg_", "") + "  |  green dashed = non-oracle VLM boxes",
                     fontsize=8)
        fig.tight_layout()
        out = os.path.join(OUTD, "t4b_%s.png" % sid)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
