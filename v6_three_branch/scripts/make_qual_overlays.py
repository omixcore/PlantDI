#!/usr/bin/env python3
"""Generate the 5 qualitative overlay panels for Fig. 6 (manuscript).

Reproducibility notes (honest, no fabrication):
- Case 1: green bbox = v4 cached VLM localization (results/mllm_prompts_cont37.json),
  coordinates already in ORIGINAL image scale (08_vlm_prompts.py divided by scale 4).
- Case 2: yellow mask = v4 color rule re-computed deterministically on CPU (no API);
          blue mask  = SAM2 re-run for VISUALIZATION with the SAME frozen v6 prompt
          pipeline (locate_with_fallback + mask_from_boxes_points). The fraction shown
          in the panel caption is the value recorded in the frozen evaluation
          (legacy37_branch3_sam2_audit.csv), not the re-run value.
- Cases 3-5: red = PlantSeg expert GT mask (data/plantseg100/masks/);
          yellow = ImageJ-style fixed-threshold baseline (eval_plantseg100.color_mask);
          blue  = SAM2 re-run for visualization (same pipeline as v6 eval);
          green = VLM bbox re-queried with the SAME frozen prompts for visualization;
          fractions in captions are the values from predictions_v6_plantseg100.csv.

Text: Arial (fallback DejaVu Sans), 7 pt, black. Output PNG at 300 dpi.
"""
import os, sys, json
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_fallback import locate_with_fallback, mask_from_boxes_points

ROOT = "."
BASE = ROOT + "/v6_three_branch"
LEG = ROOT + "/data/continuous_leaves"
PIMG = BASE + "/data/plantseg100/imgs"
PMSK = BASE + "/data/plantseg100/masks"
OUTD = BASE + "/qual_overlays"
CKPT = ROOT + "/models/sam2_hiera_tiny.pt"
os.makedirs(OUTD, exist_ok=True)
key = os.environ["DEEPSEEK_API_KEY"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "text.color": "#000000",
})

# ---------- deterministic masks (same formulas as frozen evaluation) ----------
def v4_color_lesion(bgr):
    """v4 legacy color rule (06_final_all_images.py): brown lesion / green leaf."""
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    mn = np.minimum(np.minimum(b, g), r); mx = np.maximum(np.maximum(b, g), r)
    bg = (mn > 150) & ((mx - mn) < 55)
    leaf = (g > r * 1.06) & (g > b * 1.06) & (g > 55) & (~bg)
    brown = (r > g * 0.93) & (r > b * 1.15) & (g > 30) & (~bg)
    return brown.astype(bool), leaf.astype(bool)

def ij_color_mask(bgr):
    """ImageJ-style fixed-threshold baseline (eval_plantseg100.color_mask)."""
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

def draw_bbox(ax_rgb, box, color=(0, 0.75, 0)):
    x1, y1, x2, y2 = box
    h, w = ax_rgb.shape[:2]
    ax_rgb[y1:max(y1 + 1, y2 - 1), x1:x2] = np.array(color, np.float32)
    ax_rgb[y1:y2, x1:max(x1 + 1, x2 - 1)] = np.array(color, np.float32)
    return ax_rgb

def save_panel(figpath, panels, suptitle=""):
    """panels: list of (rgb_or_path, caption). Renders one row, Arial 7pt black."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 1.85), dpi=300)
    if n == 1:
        axes = [axes]
    for ax, (img, cap) in zip(axes, panels):
        if isinstance(img, str):
            img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#000000")
        ax.set_xlabel(cap, fontsize=7, color="#000000", labelpad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=7, color="#000000", y=0.99)
    fig.tight_layout(pad=0.3)
    fig.savefig(figpath, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", figpath, flush=True)

# ---------- SAM2 (loaded once, visualization only) ----------
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", CKPT, device="cpu"))
print("SAM2 loaded", flush=True)

def sam2_mask_vis(path):
    boxes, points, psrc, tok, scale, ann = locate_with_fallback(path, key)
    bgr = cv2.imread(path)
    m = mask_from_boxes_points(sam, bgr, boxes, points)
    print("  vis SAM2 %s src=%s boxes=%d pts=%d mask_px=%d" % (
        os.path.basename(path)[:40], psrc, len(boxes), len(points), int(m.sum())), flush=True)
    return m, boxes, points, psrc

# ============================================================
# CASE 1: legacy success (geometry branch) ara_Fig1_WMA1_single
# ============================================================
sid = "ara_Fig1_WMA1_single"
p = LEG + "/" + sid + ".png"
bgr = cv2.imread(p)
ann = json.load(open(ROOT + "/results/mllm_prompts_cont37.json"))[sid]["ann"]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
rgb2 = rgb.copy()
for L in ann.get("lesions", []):
    rgb2 = draw_bbox(rgb2, L["box"], color=(0.0, 0.72, 0.0))
lb = ann.get("leaf_bbox")
if lb:
    x1, y1, x2, y2 = lb
    h, w = rgb2.shape[:2]
    rgb2[y1:y2, x1] = (0.55, 0.55, 0.55); rgb2[y1:y2, min(x2, w - 1)] = (0.55, 0.55, 0.55)
    rgb2[y1, x1:x2] = (0.55, 0.55, 0.55); rgb2[min(y2, h - 1), x1:x2] = (0.55, 0.55, 0.55)
save_panel(OUTD + "/qual_case1_legacy_success.png",
           [(rgb,  "Original\npublished 5.10 cm$^2$"),
            (rgb2, "v4 geometry branch\n(VLM bbox, pred 31.22%)")],
           suptitle="Case 1 — Legacy-37 controlled scene, pale background + isolated disc (success)")

# ============================================================
# CASE 2: legacy SAM2 failure ara_Fig7_WT_Col-0
# ============================================================
def color_severity_black(bgr):
    """EXACT v4 color rule for black-background Fig6/Fig7 (06_final_all_images.py)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = (hsv[..., 2] > 48) & (hsv[..., 1] > 35) & ~((hsv[..., 1] < 45) & (hsv[..., 2] > 205))
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    lesion = fg & (r >= g * 0.94) & (g > 45) & (b < g * 0.92)
    return fg.astype(bool), lesion.astype(bool)

sid = "ara_Fig7_WT_Col-0"
p = LEG + "/" + sid + ".png"
bgr = cv2.imread(p)
leafm, lesm = color_severity_black(bgr)
print("case2 recompute (exact v4 rule): lesion/fg = %.2f%% (CSV recorded v4 color = 92.94%%)" % (
    100.0 * lesm.sum() / max(leafm.sum(), 1)), flush=True)
m_sam, boxes, points, psrc = sam2_mask_vis(p)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
save_panel(OUTD + "/qual_case2_legacy_sam2_failure.png",
           [(rgb,               "Original\npublished 85.53%"),
            (overlay(bgr, lesm, (255, 200, 0)),  "v4 color rule\n(pred 92.94%)"),
            (overlay(bgr, m_sam, (0, 90, 255)),   "v6 SAM2 branch\n(pred 25.93%)")],
           suptitle="Case 2 — Legacy-37 dark-background leaf misrouted to SAM 2 (failure)")

# ============================================================
# CASES 3-5: PlantSeg-100 in-the-wild
# ============================================================
meta = pd.read_csv(BASE + "/data/plantseg100/meta100.csv")
pred = pd.read_csv(BASE + "/results/predictions_v6_plantseg100.csv")
pred = pred[pred.sample_id.astype(str) != "sample_id"].drop_duplicates("sample_id", keep="first")
pred = pred.set_index("sample_id")

CASES = [
    ("plantseg_bean_mosaic_virus_google_0098", "Case 3 — PlantSeg-100 green-toned mosaic (color rule fails)"),
    ("plantseg_tobacco_mosaic_virus_117",      "Case 4 — PlantSeg-100 severe green mosaic (all methods underestimate)"),
    ("plantseg_basil_downy_mildew_google_0067", "Case 5 — PlantSeg-100 green downy mildew"),
]

for sid, title in CASES:
    p = PIMG + "/" + sid + ".jpg"
    mp = PMSK + "/" + sid + ".png"
    bgr = cv2.imread(p)
    gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
    ij = ij_color_mask(bgr)
    r = pred.loc[sid]
    m_sam, boxes, points, psrc = sam2_mask_vis(p)
    ann_v = None
    # re-query VLM localization for visualization only (same frozen prompts)
    boxes_v, points_v, psrc_v, tok_v, scale_v, ann_v = locate_with_fallback(p, key)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_v = rgb.copy()
    for bx in boxes_v:
        rgb_v = draw_bbox(rgb_v, [int(round(v)) for v in bx[:4]], color=(0.0, 0.72, 0.0))
    panels = [
        (rgb,                "Original"),
        (overlay(bgr, gt, (255, 0, 0)),     "Expert GT\n%.1f%%" % (100 * float(r.published))),
        (overlay(bgr, ij, (255, 200, 0)),   "ImageJ threshold\n%.1f%%" % (100 * float(r.ij_frac))),
        (overlay(bgr, m_sam, (0, 90, 255)), "SAM 2\n%.1f%%" % (100 * float(r.sam2_frac))),
        (rgb_v,             "VLM bbox (prompt=%s)\n%.1f%%" % (str(r.prompt_source), 100 * float(r.vlm_frac))),
    ]
    save_panel(OUTD + "/qual_%s.png" % sid, panels, suptitle=title)

print("ALL OVERLAYS DONE", flush=True)
