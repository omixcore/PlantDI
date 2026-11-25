#!/usr/bin/env python3
"""Regenerate ONLY Case 2 overlay with the EXACT v4 black-bg color rule."""
import os, sys
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_fallback import locate_with_fallback, mask_from_boxes_points

ROOT = "."
BASE = ROOT + "/v6_three_branch"
LEG = ROOT + "/data/continuous_leaves"
OUTD = BASE + "/qual_overlays"
CKPT = ROOT + "/models/sam2_hiera_tiny.pt"
key = os.environ["DEEPSEEK_API_KEY"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "text.color": "#000000"})

def color_severity_black(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = (hsv[..., 2] > 48) & (hsv[..., 1] > 35) & ~((hsv[..., 1] < 45) & (hsv[..., 2] > 205))
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    lesion = fg & (r >= g * 0.94) & (g > 45) & (b < g * 0.92)
    return fg.astype(bool), lesion.astype(bool)

def overlay(bgr, mask, color):
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = mask.astype(bool)
    cm = np.array(color, np.float32) / 255.0
    out[m] = 0.55 * out[m] + 0.45 * cm
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = cm
    return np.clip(out, 0, 1)

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", CKPT, device="cpu"))
print("SAM2 loaded", flush=True)

sid = "ara_Fig7_WT_Col-0"
p = LEG + "/" + sid + ".png"
bgr = cv2.imread(p)
leafm, lesm = color_severity_black(bgr)
print("recompute lesion/fg = %.2f%% (recorded v4 color = 92.94%%)" % (
    100.0 * lesm.sum() / max(leafm.sum(), 1)), flush=True)

boxes, points, psrc, tok, scale, ann = locate_with_fallback(p, key)
m_sam = mask_from_boxes_points(sam, bgr, boxes, points)
print("vis SAM2 src=%s boxes=%d pts=%d mask_px=%d" % (psrc, len(boxes), len(points), int(m_sam.sum())), flush=True)

panels = [
    (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0, "Original\npublished 85.53%"),
    (overlay(bgr, lesm, (255, 200, 0)),  "v4 color rule\n(pred 92.94%)"),
    (overlay(bgr, m_sam, (0, 90, 255)),  "v6 SAM2 branch\n(pred 25.93%)"),
]
n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 1.85), dpi=300)
for ax, (img, cap) in zip(axes, panels):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#000000")
    ax.set_xlabel(cap, fontsize=7, color="#000000", labelpad=2)
fig.suptitle("Case 2 - Legacy-37 dark-background leaf misrouted to SAM 2 (failure)",
             fontsize=7, color="#000000", y=0.99)
fig.tight_layout(pad=0.3)
fig.savefig(OUTD + "/qual_case2_legacy_sam2_failure.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved case2 overlay", flush=True)
