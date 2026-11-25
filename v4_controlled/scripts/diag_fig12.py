#!/usr/bin/env python3
"""Diagnose Fig1/Fig2 light-background leaf cells to find a lesion rule
that matches published lesion-area ordering."""
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw

P = "./data/raw_papers/PMC9038911"
XLS = P + "/41467_2022_29788_MOESM4_ESM.xlsx"
OUT = "./data/rocole_subset/v2_probe"

CFG = [
    dict(fig="Fig1", path=P + "/41467_2022_29788_Fig1_HTML.jpg", box=(0, 18, 460, 165),
         cols=["WMA1", "KoSsPINE1-1", "KoSsPINE1-7", "ComSsPINE1-1"],
         sheet="FIg1", r0=1),
    dict(fig="Fig2", path=P + "/41467_2022_29788_Fig2_HTML.jpg", box=(0, 25, 400, 130),
         cols=["WMA1", "KoSsPG1-1", "KoSsPG1-3", "ComSsPG1-1"],
         sheet="Fig2", r0=2),
]
xls = pd.ExcelFile(XLS)


def diag(bgr):
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    mx = np.maximum(np.maximum(b, g), r); mn = np.minimum(np.minimum(b, g), r)
    # pale beige/white background: bright and low channel spread
    bg = (mn > 150) & ((mx - mn) < 55)
    # green leaf blade: green clearly dominant, not background
    leaf_green = (g > r * 1.06) & (g > b * 1.06) & (g > 55) & ~bg
    # brown lesion: red dominant or near green, blue clearly lower, mid/dark
    brown = (r > g * 0.93) & (r > b * 1.15) & (g > 30) & (~bg)
    leaf = leaf_green | brown
    return leaf, brown, bg


recs = []
vis = []
for cfg in CFG:
    im = cv2.imread(cfg["path"])
    x1, y1, x2, y2 = cfg["box"]
    panel = im[y1:y2, x1:x2]
    ph, pw = panel.shape[:2]
    df = pd.read_excel(xls, cfg["sheet"], header=None)
    for ci, col in enumerate(cfg["cols"]):
        cell = panel[:, int(ci * pw / 4):int((ci + 1) * pw / 4)]
        leaf, brown, bg = diag(cell)
        lp, bp = int(leaf.sum()), int(brown.sum())
        pc = 100.0 * bp / max(lp, 1)
        vals = pd.to_numeric(df.iloc[cfg["r0"] + ci, 1:7], errors="coerce").tolist()
        pub = float(np.nanmean(vals))
        recs.append(dict(fig=cfg["fig"], group=col, published=pub, leaf_px=lp,
                         brown_px=bp, pred=pc))
        ov = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB).copy()
        ov[leaf] = (ov[leaf] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        ov[brown] = (ov[brown] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        vis.append((cfg["fig"] + "-" + col, Image.fromarray(ov)))

d = pd.DataFrame(recs)
print(d.to_string(index=False), flush=True)
for fig, g in d.groupby("fig"):
    print(fig, "pearson=%.4f spearman=%.4f" %
          (g.published.corr(g.pred), g.published.corr(g.pred, method="spearman")), flush=True)

# contact sheet of masks
W, H = 200, 165
sh = Image.new("RGB", (W * 4, H * 2), "white")
dr = ImageDraw.Draw(sh)
for i, (lab, img) in enumerate(vis):
    img.thumbnail((W - 6, H - 22))
    x, y = (i % 4) * W + 3, (i // 4) * H + 20
    sh.paste(img, (x, y)); dr.text(((i % 4) * W + 3, (i // 4) * H + 3), lab, fill="black")
sh.save(OUT + "/fig12_mask_diag.png")
print("saved mask diag", flush=True)
