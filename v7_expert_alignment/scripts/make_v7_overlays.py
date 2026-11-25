#!/usr/bin/env python3
"""Fig.6 (v7) - qualitative overlays for Case 3/4/5 using the FROZEN v7 pipeline.

For each case: run the frozen locate_and_clip (DeepSeek polygon + leaf-mask
clipping + TTA), read the expert GT mask, and draw a panel:
  Original | Expert GT | v7 Zero-Shot predicted mask.
Percentages in captions come from the frozen per-sample prediction CSV
(v7_preds_plantseg100.csv). The predicted mask is regenerated with the SAME
frozen pipeline for visualization (the CSV stores scalars, not masks).

No ground-truth mask is ever used to build a prediction. Key is injected by
launch_with_key.py (never stored here).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(V7, "src")
sys.path.insert(0, SRC)

from polygon_and_clip import locate_and_clip  # noqa: E402

DATA = os.path.join(V7, "data", "plantseg100")
IMGS = os.path.join(DATA, "imgs")
MASKS = os.path.join(DATA, "masks")
PREDS = os.path.join(V7, "results", "v7_preds_plantseg100.csv")
OUTD = os.path.join(V7, "figures", "qual_overlays_v7")

CASES = [
    ("plantseg_bean_mosaic_virus_google_0098", "Case 3 - green-toned mosaic (colour rule fails)"),
    ("plantseg_tobacco_mosaic_virus_117", "Case 4 - severe green mosaic (under-estimated)"),
    ("plantseg_basil_downy_mildew_google_0067", "Case 5 - green downy mildew"),
]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "text.color": "#000000"})


def overlay(bgr, mask, color):
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = mask.astype(bool)
    cm = np.array(color, np.float32) / 255.0
    out[m] = 0.55 * out[m] + 0.45 * cm
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = cm
    return np.clip(out, 0, 1)


def save_panel(path, panels, suptitle=""):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.05), dpi=300)
    if n == 1:
        axes = [axes]
    for ax, (img, cap) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#000000")
        ax.set_xlabel(cap, fontsize=7, color="#000000", labelpad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=7, color="#000000", y=1.0)
    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", os.path.basename(path), flush=True)


def main():
    os.makedirs(OUTD, exist_ok=True)
    pred = pd.read_csv(PREDS)
    pred = pred[pred.sample_id.astype(str) != "sample_id"].drop_duplicates("sample_id", keep="first")
    pred = pred.set_index("sample_id")
    for sid, title in CASES:
        ip = os.path.join(IMGS, sid + ".jpg")
        mp = os.path.join(MASKS, sid + ".png")
        bgr = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        out = locate_and_clip(ip, model="deepseek-flash", reasoning="high", tta=True)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_pct = 100.0 * float(gt.sum()) / float(gt.size)
        row = pred.loc[sid]
        v7_pct = 100.0 * float(row["pred"])
        ij_pct = 100.0 * gt_pct  # placeholder unused
        # Authoritative v7 DI is the FROZEN per-sample CSV value (v7_pct).
        # The mask shown is a visualization re-run of the same frozen pipeline;
        # MLLM output is stochastic, so its coverage can differ from the CSV
        # scalar. Both are reported so the panel never misrepresents the result.
        panels = [
            (rgb, "Original"),
            (overlay(bgr, gt, (255, 0, 0)), "Expert GT\n%.1f%%" % gt_pct),
            (overlay(bgr, out["mask"], (0, 90, 255)),
             "v7 Zero-Shot (frozen CSV)\n%.1f%%  [mask re-run %.1f%%]" %
             (v7_pct, 100.0 * out["di"])),
        ]
        save_panel(os.path.join(OUTD, "qual_v7_%s.png" % sid), panels, suptitle=title)
        print("CASE %s: expert_gt=%.4f v7_csv=%.4f v7_mask_rerun=%.4f iou=%.4f dice=%.4f src=%s" % (
            sid, gt_pct / 100.0, v7_pct / 100.0, out["di"],
            float(row["iou"]), float(row["dice"]), out["prompt_source"]), flush=True)


if __name__ == "__main__":
    main()
