#!/usr/bin/env python3
"""Regenerate Case 1/3/4/5 overlays with a FIXED draw_bbox (true 2px border,
was accidentally filling wide boxes). Case 2 unchanged."""
import os, sys, json
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_fallback import locate_with_fallback

ROOT = "."
BASE = ROOT + "/v6_three_branch"
LEG = ROOT + "/data/continuous_leaves"
PIMG = BASE + "/data/plantseg100/imgs"
PMSK = BASE + "/data/plantseg100/masks"
OUTD = BASE + "/qual_overlays"
key = os.environ["DEEPSEEK_API_KEY"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "text.color": "#000000"})

def ij_color_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    fg = (hsv[..., 2] > 45) & (hsv[..., 1] > 30)
    lesion = fg & (r >= g * 0.92) & (g > 40) & (b < g * 0.95) & \
             ~((np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])) < 18)
    return lesion.astype(bool)

def overlay(bgr, mask, color):
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = mask.astype(bool)
    cm = np.array(color, np.float32) / 255.0
    out[m] = 0.55 * out[m] + 0.45 * cm
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = cm
    return np.clip(out, 0, 1)

def draw_bbox(img_f32, box, color=(0.0, 0.72, 0.0), t=2):
    """TRUE border rectangle, thickness t, clamped to image bounds."""
    h, w = img_f32.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1))); x2 = int(max(x1 + 1, min(x2, w)))
    y1 = int(max(0, min(y1, h - 1))); y2 = int(max(y1 + 1, min(y2, h)))
    c = np.array(color, np.float32)
    img_f32[y1:y1 + t, x1:x2] = c            # top
    img_f32[max(y1, y2 - t):y2, x1:x2] = c   # bottom
    img_f32[y1:y2, x1:x1 + t] = c            # left
    img_f32[y1:y2, max(x1, x2 - t):x2] = c   # right
    return img_f32

def save_panel(figpath, panels, suptitle=""):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 1.95), dpi=300)
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
    fig.savefig(figpath, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", os.path.basename(figpath), flush=True)

# ---------------- Case 1 (cached bbox, no API) ----------------
sid = "ara_Fig1_WMA1_single"
bgr = cv2.imread(LEG + "/" + sid + ".png")
ann = json.load(open(ROOT + "/results/mllm_prompts_cont37.json"))[sid]["ann"]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb2 = rgb.copy()
for L in ann.get("lesions", []):
    rgb2 = draw_bbox(rgb2, L["box"], color=(0.0, 0.72, 0.0), t=1)
lb = ann.get("leaf_bbox")
if lb:
    rgb2 = draw_bbox(rgb2, lb, color=(0.55, 0.55, 0.55), t=1)
save_panel(OUTD + "/qual_case1_legacy_success.png",
           [(rgb,  "Original\npublished 5.10 cm$^2$"),
            (rgb2, "v4 geometry branch\n(VLM bbox, pred 31.22%)")],
           suptitle="Case 1 - Legacy-37 controlled scene, pale background + isolated disc (success)")

# ---------------- Cases 3-5 (VLM re-query for visualization) ----------------
pred = pd.read_csv(BASE + "/results/predictions_v6_plantseg100.csv")
pred = pred[pred.sample_id.astype(str) != "sample_id"].drop_duplicates("sample_id", keep="first").set_index("sample_id")

CASES = [
    ("plantseg_bean_mosaic_virus_google_0098", "Case 3 - PlantSeg-100 green-toned mosaic (color rule fails)"),
    ("plantseg_tobacco_mosaic_virus_117",      "Case 4 - PlantSeg-100 severe green mosaic (all methods underestimate)"),
    ("plantseg_basil_downy_mildew_google_0067", "Case 5 - PlantSeg-100 green downy mildew"),
]
for sid, title in CASES:
    p = PIMG + "/" + sid + ".jpg"
    bgr = cv2.imread(p)
    gt = cv2.imread(PMSK + "/" + sid + ".png", cv2.IMREAD_GRAYSCALE) > 0
    ij = ij_color_mask(bgr)
    r = pred.loc[sid]
    boxes_v, points_v, psrc_v, tok_v, scale_v, ann_v = locate_with_fallback(p, key)
    print("%s vis-query src=%s boxes=%d" % (sid[:44], psrc_v, len(boxes_v)), flush=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_v = rgb.copy()
    for bx in boxes_v:
        rgb_v = draw_bbox(rgb_v, [int(round(v)) for v in bx[:4]], color=(0.0, 0.72, 0.0), t=2)
    # SAM2 visualization mask: re-run with same frozen pipeline
    from sam2_fallback import mask_from_boxes_points
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", ROOT + "/models/sam2_hiera_tiny.pt", device="cpu"))
    m_sam = mask_from_boxes_points(sam, bgr, boxes_v, points_v)
    panels = [
        (rgb,  "Original"),
        (overlay(bgr, gt, (255, 0, 0)),     "Expert GT\n%.1f%%" % (100 * float(r.published))),
        (overlay(bgr, ij, (255, 200, 0)),   "ImageJ threshold\n%.1f%%" % (100 * float(r.ij_frac))),
        (overlay(bgr, m_sam, (0, 90, 255)), "SAM 2\n%.1f%%" % (100 * float(r.sam2_frac))),
        (rgb_v, "VLM bbox (%s)\n%.1f%%" % (str(r.prompt_source), 100 * float(r.vlm_frac))),
    ]
    save_panel(OUTD + "/qual_%s.png" % sid, panels, suptitle=title)

print("ALL FIXED OVERLAYS DONE", flush=True)
