#!/usr/bin/env python3
"""Probe continuous ground truth: Fig6/Fig7 detached leaves on black background.

Per group, 3 replicate leaves are segmented from black background; lesion
pixels are brown/necrotic (R >= G) vs healthy green (G >> R). White inoculation
discs (low saturation, very high value) are excluded. The group-level mean
lesion area percentage is compared with the published mean lesion area cm2.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

R = "."
P = R + "/data/raw_papers/PMC9038911"
XL = P + "/41467_2022_29788_MOESM4_ESM.xlsx"
OUT = R + "/data/rocole_subset/v2_probe"
os.makedirs(OUT, exist_ok=True)

# (figure, panel box in original coords, n_rows, n_cols, sheet, first_data_row)
FIGS = [
    dict(fig="Fig1", path=P + "/41467_2022_29788_Fig1_HTML.jpg",
         box=(0, 18, 460, 165), rows=["single"],
         cols=["WMA1", "KoSsPINE1-1", "KoSsPINE1-7", "ComSsPINE1-1"],
         sheet="FIg1", r0=1, nrows=4, nrow=1, order="col", nrep=1),
    dict(fig="Fig2", path=P + "/41467_2022_29788_Fig2_HTML.jpg",
         box=(0, 25, 400, 130), rows=["single"],
         cols=["WMA1", "KoSsPG1-1", "KoSsPG1-3", "ComSsPG1-1"],
         sheet="Fig2", r0=2, nrows=4, nrow=1, order="col", nrep=1),
    dict(fig="Fig6", path=P + "/41467_2022_29788_Fig6_HTML.jpg",
         box=(14, 30, 448, 468), rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WMA1", "KoSsPINE1-7", "KoSsPG1-2"], sheet="Fig6", r0=2, nrows=9, nrow=3, order="col", nrep=3),
    dict(fig="Fig7", path=P + "/41467_2022_29788_Fig7_HTML.jpg",
         box=(70, 20, 500, 660), rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WT", "KoBcPINE1-6"], sheet="Fig7", r0=2, nrows=6, nrow=3, order="row", nrep=3),
]

xls = pd.ExcelFile(XL)

def leaf_lesion_pct(bgr):
    """One detached leaf crop -> (leaf_px, lesion_pct)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # black background: value < 48. White discs: saturation < 45 and value > 215
    fg = (hsv[..., 2] > 48) & (hsv[..., 1] > 35) & ~((hsv[..., 1] < 45) & (hsv[..., 2] > 205))
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    # necrotic / chlorotic lesion: red channel approaches/exceeds green.
    lesion = fg & (r >= g * 0.94) & (g > 45) & (b < g * 0.92)
    return int(fg.sum()), 100.0 * lesion.sum() / max(int(fg.sum()), 1)

records = []
sheets = []
for cfg in FIGS:
    im = cv2.imread(cfg["path"])
    x1, y1, x2, y2 = cfg["box"]
    panel = im[y1:y2, x1:x2].copy()
    ph, pw = panel.shape[:2]
    ncol, nrow = len(cfg["cols"]), len(cfg["rows"])
    cell_w, cell_h = pw / ncol, ph / nrow
    df = pd.read_excel(xls, cfg["sheet"], header=None)
    for ri, row in enumerate(cfg["rows"]):
        for ci, col in enumerate(cfg["cols"]):
            label = "%s-%s" % (col, row)
            idx = (ci * cfg["nrow"] + ri) if cfg["order"] == "col" else (ri * ncol + ci)
            vals = pd.to_numeric(df.iloc[cfg["r0"] + idx, 1:7], errors="coerce").tolist()
            pub = float(np.nanmean(vals))
            cx1 = int(ci * cell_w); cx2 = int((ci + 1) * cell_w)
            cy1 = int(ri * cell_h); cy2 = int((ri + 1) * cell_h)
            cell = panel[cy1:cy2, cx1:cx2]
            # split the cell into 3 vertical strips: one leaf per replicate
            ch, cw = cell.shape[:2]
            pcts, lpxs = [], []
            for k in range(cfg["nrep"]):
                strip = cell[:, int(k * cw / 3):int((k + 1) * cw / 3)]
                lp, pc = leaf_lesion_pct(strip)
                if lp > 250:
                    pcts.append(pc); lpxs.append(lp)
            pred = float(np.mean(pcts)) if pcts else 0.0
            records.append(dict(sample_id="%s_%s_%s" % (cfg["fig"], col, row.replace(":", "")),
                                crop="arabidopsis", fig=cfg["fig"], group=label,
                                published=pub, pred=pred, n_leaf=len(pcts),
                                preds=";".join("%.2f" % v for v in pcts)))
            sheets.append((cfg["fig"], label, cell))

# sheet of cells for visual QC
thumbs = []
for fig, label, cell in sheets:
    vis = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    vis = Image.fromarray(vis); vis.thumbnail((150, 150))
    c = Image.new("RGB", (160, 175), "white"); c.paste(vis, (5, 20))
    ImageDraw.Draw(c).text((4, 4), label[:24], fill="black")
    thumbs.append(c)
if thumbs:
    sh = Image.new("RGB", (160 * 5, 175 * ((len(thumbs) + 4) // 5)), "white")
    for i, t in enumerate(thumbs):
        sh.paste(t, ((i % 5) * 160, (i // 5) * 175))
    sh.save(OUT + "/v2_fig_cells.png")

df = pd.DataFrame(records)
df.to_csv(R + "/results/v2_probe_figures.csv", index=False)
print(df.to_string(index=False), flush=True)
print("\nCorrelations:", flush=True)
for fig, g in df.groupby("fig"):
    if len(g) > 2 and g.pred.nunique() > 1:
        print(fig, "n=%d pearson=%.3f spearman=%.3f" %
              (len(g), g.published.corr(g.pred), g.published.corr(g.pred, method="spearman")), flush=True)
print("ALL pearson=%.3f spearman=%.3f" %
      (df.published.corr(df.pred), df.published.corr(df.pred, method="spearman")), flush=True)
# cross-batch normalized pooled correlation (within-figure z-scores)
df["z_pub"] = df.groupby("fig").published.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
df["z_pred"] = df.groupby("fig").pred.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
df.to_csv(R + "/results/v2_probe_figures.csv", index=False)
print("POOLED within-figure standardized: pearson=%.4f spearman=%.4f n=%d" %
      (df.z_pub.corr(df.z_pred), df.z_pub.corr(df.z_pred, method="spearman"), len(df)), flush=True)
core = df[df.fig.isin(["Fig6", "Fig7"])]
cp = (core.z_pub - core.z_pub.mean()) / core.z_pub.std(ddof=0)
cq = (core.z_pred - core.z_pred.mean()) / core.z_pred.std(ddof=0)
print("CORE Fig6+Fig7 pooled: pearson=%.4f spearman=%.4f n=%d" %
      (cp.corr(cq), cp.corr(cq, method="spearman"), len(core)), flush=True)
