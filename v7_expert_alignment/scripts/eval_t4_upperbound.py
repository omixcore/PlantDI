#!/usr/bin/env python3
"""T4 evaluation - Supervised Domain-Adapted Upper Bound on PlantSeg-100.

PROMPT PROTOCOL (must be reported explicitly, NOT comparable to zero-shot):
  The box prompt is the ORACLE bounding box derived from the Expert GT mask
  (min/max x,y of foreground). This deliberately removes localization error
  to isolate the pixel-level mask quality of the SAM2 decoder. Because it uses
  test-set GT, every number here is an OPTIMISTIC UPPER BOUND and is labeled
  `oracle_box` in the summary. It is NOT a deployment result and NOT directly
  comparable to v6/v7 zero-shot rows.

Two arms are evaluated with the SAME oracle boxes:
  * sam2_ft_oracle : decoder weights from T4 fine-tuning
  * sam2_base_oracle : original SAM2 tiny decoder (control)

Metrics: r / Spearman rho / Lin's CCC / MAE on DI (raw scale, n=100),
plus mIoU / Dice on binary masks. All computed against frozen meta100 published.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metrics import summarize  # noqa: E402

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def oracle_box(gt: np.ndarray):
    ys, xs = np.where(gt)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def build_model(sam2_root: str, ckpt: str, decoder_state=None, device="cuda"):
    model = build_sam2(config_file="configs/sam2/sam2_hiera_t.yaml",
                       ckpt_path=ckpt, device=device, mode="eval")
    if decoder_state is not None:
        model.sam_mask_decoder.load_state_dict(decoder_state["sam_mask_decoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return SAM2ImagePredictor(model)


@torch.no_grad()
def predict_mask(predictor, image_rgb, box):
    predictor.set_image(image_rgb)
    feats = predictor._features
    x1, y1, x2, y2 = box
    coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32,
                          device=predictor.device)
    labels = torch.tensor([[2, 3]], dtype=torch.int64, device=predictor.device)
    sparse, dense = predictor.model.sam_prompt_encoder(
        points=(coords, labels), boxes=None, masks=None)
    high = [f[0:1] for f in feats["high_res_feats"]]
    low, _, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=feats["image_embed"][0:1],
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high,
    )
    masks = predictor._transforms.postprocess_masks(low, predictor._orig_hw[0])
    return (masks[0, 0] > 0).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ft-weights", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--maskdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = {}
    with open(args.meta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["sample_id"]] = float(r["published"])

    ft_state = torch.load(args.ft_weights, map_location="cpu", weights_only=True)
    predictors = {
        "sam2_base_oracle": build_model(args.sam2_root, args.checkpoint, None, args.device),
        "sam2_ft_oracle": build_model(args.sam2_root, args.checkpoint, ft_state, args.device),
    }

    preds = {k: [] for k in predictors}
    detail = []
    t0 = time.time()
    ids = sorted(meta)
    for i, sid in enumerate(ids):
        ip = os.path.join(args.imgdir, sid + ".jpg")
        mp = os.path.join(args.maskdir, sid + ".png")
        bgr = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        if bgr is None or gt.size == 0:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = gt.shape
        box = oracle_box(gt)
        pub = meta[sid]
        rec = {"sample_id": sid, "published": pub, "gt_ratio": float(gt.mean())}
        for name, predictor in predictors.items():
            if box is None:
                pm = np.zeros_like(gt)
            else:
                pm = predict_mask(predictor, rgb, box)
                if pm.shape != gt.shape:
                    pm = cv2.resize(pm.astype(np.float32), (w, h)) > 0.5
            inter = np.logical_and(pm, gt).sum()
            union = np.logical_or(pm, gt).sum()
            iou = float(inter) / union if union else 0.0
            dice = float(2 * inter) / (pm.sum() + gt.sum()) if (pm.sum() + gt.sum()) else 0.0
            pred_frac = float(pm.mean())
            preds[name].append({"published": pub, "pred": pred_frac,
                                "iou": iou, "dice": dice})
            rec[name + "_pred"] = pred_frac
            rec[name + "_iou"] = iou
            rec[name + "_dice"] = dice
        detail.append(rec)
        if (i + 1) % 20 == 0:
            print("eval %d/%d  %.1fs" % (i + 1, len(ids), time.time() - t0), flush=True)

    summary = []
    for name, rows in preds.items():
        s = summarize([r["published"] for r in rows], [r["pred"] for r in rows],
                      [r["iou"] for r in rows], [r["dice"] for r in rows])
        summary.append({
            "domain": "plantseg100_raw",
            "method": name,
            "paradigm": "Supervised Domain-Adapted Upper Bound (ORACLE box prompt)",
            "status": "COMPUTED",
            "n": s["n"], "r": s["r"], "rho": s["rho"], "ccc": s["ccc"],
            "mae": s["mae"], "p": s["p"], "mIoU": s["mean_iou"], "dice": s["mean_dice"],
            "prompt_protocol": "oracle_gt_bbox (test GT used; NOT comparable to zero-shot)",
            "source": os.path.abspath(args.ft_weights),
        })
        print(name, json.dumps({k: summary[-1][k] for k in ("n", "r", "ccc", "mae", "mIoU", "dice")}), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    dpath = os.path.splitext(args.out)[0] + "_detail.json"
    with open(dpath, "w", encoding="utf-8") as fh:
        json.dump({"protocol": "oracle_gt_bbox_upper_bound", "n": len(detail),
                   "elapsed_s": round(time.time() - t0, 1), "detail": detail}, fh)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
