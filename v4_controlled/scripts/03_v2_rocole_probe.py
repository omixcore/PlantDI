#!/usr/bin/env python3
"""Probe: official RoCoLe leaf mask + color-rule rust segmentation.

Calibration is visual/rule-based (no fitting to ground-truth labels):
healthy coffee leaf is green (Lab a* below neutral), rust pustules are
yellow-orange/brown (a* above neutral), specular highlights have low
saturation / very high value.
"""
import json, os, re
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = "."
ROC = os.path.join(ROOT, "data/rocole_subset")
OUT = os.path.join(ROOT, "data/rocole_subset/v2_probe")
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(os.path.join(ROC, "rocole-coco.json"), encoding="utf-8"))
man = pd.read_csv(os.path.join(ROC, "rocole_subset_manifest.csv"))

# map basename -> annotation
ann_by_name = {}
for a in coco["annotations"]:
    if a["category_id"] != 1:
        continue
    img = next(i for i in coco["images"] if i["id"] == a["image_id"])
    base = img["file_name"].rsplit("/", 1)[-1]
    m = re.search(r"(C\d+P\d+[EH]\d+\.jpg)$", base, re.I)
    if m:
        a = dict(a)
        a["_coco_w"] = img.get("width", 2048)
        a["_coco_h"] = img.get("height", 1152)
        ann_by_name[m.group(1)] = a
print("coco leaf anns:", len(ann_by_name), flush=True)

rows = []
for fname, ann in ann_by_name.items():
    if fname not in set(man["filename"]):
        continue
    lvl = int(man.loc[man["filename"] == fname, "severity_level"].iloc[0])
    sid = "rocole_%02d_%s" % (lvl, fname[:-4])
    bgr = cv2.imread(os.path.join(ROC, fname))
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    sx, sy = w / float(ann.get("_coco_w", 2048)), h / float(ann.get("_coco_h", 1152))
    for poly in ann["segmentation"]:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    a_ch, b_ch = lab[..., 1].astype(np.int16), lab[..., 2].astype(np.int16)
    s_ch, v_ch = hsv[..., 1], hsv[..., 2]
    leaf = mask > 0
    # rust rule: reddish-yellow (a above neutral), yellow (b above neutral),
    # saturated, not specular white
    rust = ((a_ch >= 134) & (b_ch >= 128) & (s_ch >= 45) & (v_ch <= 248)) & leaf
    # remove tiny salt-and-pepper noise, keep pustules >= 2 px
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rust.astype(np.uint8), 8)
    clean = np.zeros_like(rust)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 2:
            clean |= labels == i
    pct = 100.0 * clean.sum() / max(int(leaf.sum()), 1)
    rows.append((sid, lvl, int(leaf.sum()), int(clean.sum()), round(pct, 3),
                 float(np.mean(a_ch[leaf])), float(np.median(a_ch[leaf])),
                 float(np.mean(b_ch[leaf]))))
    if len(rows) % 10 == 0:
        print("probe", len(rows), flush=True)

df = pd.DataFrame(rows, columns=["sample_id", "level", "leaf_px", "rust_px", "rust_pct",
                                 "a_mean", "a_med", "b_mean"])
df.to_csv(os.path.join(ROOT, "results/v2_probe_rocole.csv"), index=False)
print("matched samples:", len(df), flush=True)
print(df.groupby("level").rust_pct.agg(["size", "mean", "min", "median", "max"]).round(2).to_string(), flush=True)
print("spearman:", df.level.corr(df.rust_pct, method="spearman").round(3), flush=True)

# visual contact sheet for 8 samples using a fixed a* threshold overlay
sel = df.groupby("level", group_keys=False).head(2)["sample_id"].tolist()
ims = []
for sid in sel:
    row = df[df.sample_id == sid].iloc[0]
    fname = sid.split("_", 2)[2] + ".jpg"
    bgr = cv2.imread(os.path.join(ROC, fname))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ann = ann_by_name[fname]
    mask = np.zeros(bgr.shape[:2], np.uint8)
    sx = bgr.shape[1] / float(ann["_coco_w"]); sy = bgr.shape[0] / float(ann["_coco_h"])
    for poly in ann["segmentation"]:
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        pts[:, 0] *= sx; pts[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rust = ((lab[..., 1].astype(np.int16) >= 134) & (lab[..., 2].astype(np.int16) >= 128)
            & (hsv[..., 1] >= 45) & (hsv[..., 2] <= 248)) & (mask > 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb = rgb * 0.55 + 255 * 0.45 * (mask[..., None] > 0)
    rgb = rgb * (1 - rust[..., None] * 0.6) + np.array([255, 30, 30]) * rust[..., None] * 0.6
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    im.thumbnail((420, 260))
    canvas = Image.new("RGB", (440, 290), "white")
    canvas.paste(im, (10, 25))
    ImageDraw.Draw(canvas).text((10, 6), "%s lvl=%d pct=%.2f" % (sid, row.level, row.rust_pct), fill="black")
    ims.append(canvas)
sheet = Image.new("RGB", (440 * 2, 290 * ((len(ims) + 1) // 2)), "white")
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 2) * 440, (i // 2) * 290))
sheet.save(os.path.join(OUT, "rocole_color_probe.png"))
print("saved probe sheet", flush=True)
