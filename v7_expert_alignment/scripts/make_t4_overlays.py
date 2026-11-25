#!/usr/bin/env python3
"""T4 qualitative overlays for the three key cases (oracle box prompt).

Three panels: Original | Expert GT | FT-SAM2 prediction. Caption numbers come
from the frozen detail JSON (t4_upperbound_detail.json), not a fresh re-run.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_t4_upperbound import build_model, predict_mask, oracle_box  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 7,
                     "svg.fonttype": "none", "text.color": "black"})

CASES = ["plantseg_bean_mosaic_virus_google_0098",
         "plantseg_tobacco_mosaic_virus_117",
         "plantseg_basil_downy_mildew_google_0067"]
IMG = os.path.join(ROOT, "data", "plantseg100", "imgs")
MSK = os.path.join(ROOT, "data", "plantseg100", "masks")
OUTD = os.path.join(ROOT, "figures", "qual_overlays_t4")
CKPT = os.path.join(ROOT, "weights", "sam2_hiera_tiny.pt")
FT = os.path.join(ROOT, "weights", "sam2_tiny_decoder_ft.pt")
SAM2ROOT = os.path.join(ROOT, "third_party", "sam2")
DETAIL = os.path.join(ROOT, "results", "t4_upperbound_detail.json")


def overlay(bgr, mask, color):
    out = bgr.copy()
    m = mask.astype(bool)
    layer = np.zeros_like(bgr); layer[:] = color
    out[m] = (0.45 * out[m] + 0.55 * layer[m]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(OUTD, exist_ok=True)
    detail = {r["sample_id"]: r for r in json.load(open(DETAIL, encoding="utf-8"))["detail"]}
    ft_state = torch.load(FT, map_location="cpu", weights_only=True)
    pred = build_model(SAM2ROOT, CKPT, ft_state, "cuda")
    for sid in CASES:
        bgr = cv2.imread(os.path.join(IMG, sid + ".jpg"))
        gt = cv2.imread(os.path.join(MSK, sid + ".png"), cv2.IMREAD_GRAYSCALE) > 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = oracle_box(gt)
        pm = predict_mask(pred, rgb, box)
        if pm.shape != gt.shape:
            pm = cv2.resize(pm.astype(np.float32), (gt.shape[1], gt.shape[0])) > 0.5
        d = detail[sid]
        panels = [
            (rgb, "Original"),
            (overlay(bgr, gt, (0, 0, 255)), "Expert GT\n%.1f%%" % (100 * d["published"])),
            (overlay(bgr, pm, (255, 120, 0)),
             "FT-SAM2 (oracle box)\nDI=%.1f%%  IoU=%.2f\n(detail JSON; not a fresh re-run)" %
             (100 * d["sam2_ft_oracle_pred"], d["sam2_ft_oracle_iou"])),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
        for ax, (img, title) in zip(axs, panels):
            ax.imshow(img); ax.set_title(title, fontsize=7); ax.axis("off")
        fig.suptitle(sid.replace("plantseg_", ""), fontsize=8)
        fig.tight_layout()
        out = os.path.join(OUTD, "t4_%s.png" % sid)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("saved", out, "GT=%.4f FT=%.4f IoU=%.4f" %
              (d["published"], d["sam2_ft_oracle_pred"], d["sam2_ft_oracle_iou"]), flush=True)


if __name__ == "__main__":
    main()
