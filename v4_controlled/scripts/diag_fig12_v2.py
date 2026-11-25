#!/usr/bin/env python3
"""Fig1/Fig2: lesion-anchored region growing. Detect brown inoculation disc,
grow green blade around it; no fixed cell grid needed."""
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw

P = "./data/raw_papers/PMC9038911"
XLS = P + "/41467_2022_29788_MOESM4_ESM.xlsx"
OUT = "./data/rocole_subset/v2_probe"
CFG = [
    dict(fig="Fig1", path=P + "/41467_2022_29788_Fig1_HTML.jpg", box=(0, 18, 460, 165),
         cols=["WMA1", "KoSsPINE1-1", "KoSsPINE1-7", "ComSsPINE1-1"], sheet="FIg1", r0=1),
    dict(fig="Fig2", path=P + "/41467_2022_29788_Fig2_HTML.jpg", box=(0, 25, 400, 130),
         cols=["WMA1", "KoSsPG1-1", "KoSsPG1-3", "ComSsPG1-1"], sheet="Fig2", r0=2),
]


def segment(bgr):
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    mx = np.maximum(np.maximum(b, g), r); mn = np.minimum(np.minimum(b, g), r)
    bg = (mn > 150) & ((mx - mn) < 55)
    green = (g > r * 1.06) & (g > b * 1.06) & (g > 55) & ~bg
    brown = (r > g * 0.93) & (r > b * 1.15) & (g > 30) & ~bg
    # clean brown and find connected discs
    br = (brown.astype(np.uint8) * 255)
    br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(br, 8)
    discs = [(stats[i, cv2.CC_STAT_AREA], cents[i], i)
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 160]
    return green, brown, br, discs, lab


vis = []
for cfg in CFG:
    im = cv2.imread(cfg["path"])
    x1, y1, x2, y2 = cfg["box"]
    panel = im[y1:y2, x1:x2].copy()
    green, brown, br, discs, lab = segment(panel)
    print(cfg["fig"], "discs found:", [(round(a), tuple(np.round(c).astype(int))) for a, c, i in discs], flush=True)
    ov = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    for a, c, i in discs:
        cv2.circle(ov, tuple(np.round(c).astype(int)), 3, (0, 0, 255), -1)
    vis.append((cfg["fig"], Image.fromarray(ov)))

W = 460
sh = Image.new("RGB", (W, 330), "white"); dr = ImageDraw.Draw(sh)
for i, (lab, img) in enumerate(vis):
    img.thumbnail((W - 6, 160))
    sh.paste(img, (3, i * 165 + 18)); dr.text((3, i * 165 + 3), lab, fill="black")
sh.save(OUT + "/fig12_disc_detect.png")
print("saved", flush=True)
