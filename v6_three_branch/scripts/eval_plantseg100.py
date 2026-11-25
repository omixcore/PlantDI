#!/usr/bin/env python3
"""V6 core evaluation on PlantSeg-100 (raw scale, NO z-score).

Four baselines + frozen v6 three-branch agent:
  1 ImageJ fixed threshold (v5 frozen color rule)
  2 Pure VLM bbox (rasterized largest lesion box)
  3 Pure SAM2 (with two-level prompt fallback, prompt_source recorded)
  4 V6 frozen agent: branch from frozen_policy_v6
       Branch 1 color -> color mask
       Branch 2 geometry -> VLM bbox mask
       Branch 3 sam2   -> SAM2 mask (fallback-enabled)

Ground truth = official expert mask (lesion pixels / whole image).
All methods share the SAME localization calls (cached) for fairness.
Resumable: writes CSV row-by-row; re-run skips done sample_ids.
"""
import os, sys, json, time
import numpy as np, pandas as pd, cv2
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from frozen_policy_v6 import route as frozen_route, policy_hash
from router_brain_v6 import query as route_query
from sam2_fallback import (locate_with_fallback, mask_from_boxes_points,
                           PRIMARY_SYSTEM, _call as vlm_call, parse as jparse,
                           _encode_small)

BASE = "./v6_three_branch"
IMG = BASE + "/data/plantseg100/imgs"
MSK = BASE + "/data/plantseg100/masks"
META = BASE + "/data/plantseg100/meta100.csv"
OUT = BASE + "/results/predictions_v6_plantseg100.csv"
CKPT = "./models/sam2_hiera_tiny.pt"
key = os.environ["DEEPSEEK_API_KEY"]


def color_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    fg = (hsv[..., 2] > 45) & (hsv[..., 1] > 30)
    lesion = fg & (r >= g * 0.92) & (g > 40) & (b < g * 0.95) & \
             ~((np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])) < 18)
    return lesion


def metrics(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = (p & g).sum(); union = (p | g).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return iou, dice


def ccc(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]; y = np.asarray(y)[ok]
    if len(x) < 2:
        return float("nan")
    return 2 * np.cov(x, y)[0, 1] / (np.var(x, ddof=1) + np.var(y, ddof=1) + (np.mean(x) - np.mean(y)) ** 2)


def vlm_mask_from_boxes(shape, boxes):
    """Rasterize lesion boxes (already in ORIGINAL image coordinates)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for bb in boxes:
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in bb[:4]]
        except Exception:
            continue
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m.astype(bool)


def main():
    print("FROZEN POLICY SHA256:", policy_hash(), flush=True)
    meta = pd.read_csv(META)
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT)
        # drop repeated header rows and duplicate sample_ids (keep first) from any interrupted run
        d0 = d0[d0.sample_id.astype(str) != "sample_id"]
        d0 = d0.drop_duplicates(subset="sample_id", keep="first")
        done = set(d0.sample_id.tolist())
        d0.to_csv(OUT, index=False)  # rewrite clean
        print("resuming, done =", len(done), flush=True)
    write_header = not os.path.exists(OUT)

    sam = None
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", CKPT, device="cpu"))
        print("SAM2 loaded", flush=True)
    except Exception as e:
        print("SAM2 unavailable:", repr(e), flush=True)

    for i, r in meta.iterrows():
        sid = r.sample_id
        if sid in done:
            continue
        t0 = time.time()
        p = IMG + "/" + sid + ".jpg"
        mp = MSK + "/" + sid + ".png"
        bgr = cv2.imread(p)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        gt_frac = float(gt.sum()) / float(gt.size)

        # --- routing brain (v6, image only) ---
        dec, tok_route = route_query(p, key)
        if dec is None:
            dec = {"background": "natural", "scene_type": "in_the_wild",
                   "lesion_color": "non_brown_or_mixed", "lesion_morph": "diffuse_necrosis",
                   "rationale": "parse-fail -> conservative SAM2"}
        branch, why = frozen_route(dec)

        # --- shared localization (primary + fallback) ---
        boxes, points, psrc, tok_loc, scale, ann_raw = locate_with_fallback(p, key)

        # method 1 ImageJ
        cm = color_mask(bgr)
        ij_frac = float(cm.sum()) / float(cm.size)
        ij_iou, ij_dice = metrics(cm, gt)
        # method 2 Pure VLM (rasterize boxes from the SAME localization ann)
        vm = vlm_mask_from_boxes(bgr.shape[:2], boxes)
        vlm_frac = float(vm.sum()) / float(vm.size)
        vlm_iou, vlm_dice = metrics(vm, gt)
        # method 3 Pure SAM2 (same prompt; fallback if empty)
        sm = mask_from_boxes_points(sam, bgr, boxes, points) if sam else np.zeros_like(cm)
        sam_frac = float(sm.sum()) / float(sm.size)
        sam_iou, sam_dice = metrics(sm, gt)
        # method 4 frozen v6 agent
        if branch == "color":
            ag = cm
        elif branch == "geometry":
            ag = vm
        else:
            ag = sm
        ag_frac = float(ag.sum()) / float(ag.size)
        ag_iou, ag_dice = metrics(ag, gt)

        row = dict(sample_id=sid, crop=r.crop, disease=r.disease, in_v5_48=int(r.in_v5_48),
                   published=gt_frac,
                   ij_frac=ij_frac, vlm_frac=vlm_frac, sam2_frac=sam_frac, agent_frac=ag_frac,
                   ij_iou=ij_iou, ij_dice=ij_dice, vlm_iou=vlm_iou, vlm_dice=vlm_dice,
                   sam2_iou=sam_iou, sam2_dice=sam_dice, agent_iou=ag_iou, agent_dice=ag_dice,
                   branch=branch, route_scene=dec.get("scene_type", ""),
                   route_color=dec.get("lesion_color", ""), route_bg=dec.get("background", ""),
                   route_morph=dec.get("lesion_morph", ""), route_reason=why,
                   prompt_source=psrc, n_boxes=len(boxes), n_points=len(points),
                   tok_route=tok_route, tok_loc=tok_loc, seconds=round(time.time() - t0, 2))
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=write_header, index=False)
        write_header = False
        print("[%d/%d] %-36s scene=%-11s branch=%-8s src=%-8s gt=%.3f ag=%.3f ij=%.3f vlm=%.3f sam=%.3f %.1fs" % (
            i + 1, len(meta), sid[:36], dec.get("scene_type", "?"), branch, psrc,
            gt_frac, ag_frac, ij_frac, vlm_frac, sam_frac, time.time() - t0), flush=True)

    d = pd.read_csv(OUT)
    print("\n=== PlantSeg-100 raw-scale scoring (NO z-score) ===", flush=True)
    def score(col, name):
        ok = np.isfinite(d[col])
        r, p = stats.pearsonr(d.published[ok], d[col][ok])
        rho, _ = stats.spearmanr(d.published[ok], d[col][ok])
        cc = ccc(d.published[ok].values, d[col][ok].values)
        mae = float((d.published[ok] - d[col][ok]).abs().mean())
        print("%-22s r=%+.4f rho=%+.4f CCC=%+.4f MAE=%.4f n=%d" % (name, r, rho, cc, mae, ok.sum()))
    for c, n in [("ij_frac", "ImageJ fixed thr"), ("vlm_frac", "Pure VLM bbox"),
                 ("sam2_frac", "Pure SAM2+fallback"), ("agent_frac", "V6 frozen agent")]:
        score(c, n)
    print("\n=== pixel metrics ===", flush=True)
    for m in ["ij", "vlm", "sam2", "agent"]:
        print("%-6s mIoU=%.4f Dice=%.4f" % (m, d[m + "_iou"].mean(), d[m + "_dice"].mean()))
    print("branch:", d.branch.value_counts().to_dict())
    print("scene:", d.route_scene.value_counts().to_dict())
    print("prompt_source:", d.prompt_source.value_counts().to_dict())
    print("mean seconds/img=%.2f" % d.seconds.mean())


if __name__ == "__main__":
    main()
