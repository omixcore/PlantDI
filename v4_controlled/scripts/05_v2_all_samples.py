#!/usr/bin/env python3
"""V2 unified pipeline: color-based lesion quantification with published continuous truth.

Generates results/predictions_v2.csv for all samples with continuous truth:
  - arabidopsis Fig1/Fig2/Fig6/Fig7 (lesion area cm2, Nat Commun 2022)
  - rice Fig1 (disease index %, Nat Commun 2024)
  - mango Figure S11 (lesion area %, digitized)
  - RoCoLe (ordinal rust level, official OIRSA grading) -- validation only
"""
import os, re, json
import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = "."
RAW = ROOT + "/data/raw_papers"
LEAVES = ROOT + "/data/processed_leaves"
ROC = ROOT + "/data/rocole_subset"
OUT = ROOT + "/results"
os.makedirs(OUT, exist_ok=True)

recs = []

def color_severity(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = (hsv[..., 2] > 48) & (hsv[..., 1] > 35) & ~((hsv[..., 1] < 45) & (hsv[..., 2] > 205))
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    lesion = fg & (r >= g * 0.94) & (g > 45) & (b < g * 0.92)
    lp = int(fg.sum())
    return lp, int(lesion.sum()), 100.0 * lesion.sum() / max(lp, 1)

# ---------- Arabidopsis figures ----------
P = RAW + "/PMC9038911"
XL = P + "/41467_2022_29788_MOESM4_ESM.xlsx"
CFG = [
    dict(fig="Fig1", path=P + "/41467_2022_29788_Fig1_HTML.jpg", box=(0, 18, 460, 165),
         rows=["single"], cols=["WMA1", "KoSsPINE1-1", "KoSsPINE1-7", "ComSsPINE1-1"],
         sheet="FIg1", r0=1, nrow=1, order="col", nrep=1),
    dict(fig="Fig2", path=P + "/41467_2022_29788_Fig2_HTML.jpg", box=(0, 25, 400, 130),
         rows=["single"], cols=["WMA1", "KoSsPG1-1", "KoSsPG1-3", "ComSsPG1-1"],
         sheet="Fig2", r0=2, nrow=1, order="col", nrep=1),
    dict(fig="Fig6", path=P + "/41467_2022_29788_Fig6_HTML.jpg", box=(14, 30, 448, 468),
         rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WMA1", "KoSsPINE1-7", "KoSsPG1-2"],
         sheet="Fig6", r0=2, nrow=3, order="col", nrep=3),
    dict(fig="Fig7", path=P + "/41467_2022_29788_Fig7_HTML.jpg", box=(70, 20, 500, 660),
         rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WT", "KoBcPINE1-6"], sheet="Fig7", r0=2, nrow=3, order="row", nrep=3),
]
xls = pd.ExcelFile(XL)
for cfg in CFG:
    im = cv2.imread(cfg["path"])
    x1, y1, x2, y2 = cfg["box"]
    panel = im[y1:y2, x1:x2]
    ph, pw = panel.shape[:2]
    nc, nr = len(cfg["cols"]), len(cfg["rows"])
    df = pd.read_excel(xls, cfg["sheet"], header=None)
    for ri, row in enumerate(cfg["rows"]):
        for ci, col in enumerate(cfg["cols"]):
            idx = (ci * cfg["nrow"] + ri) if cfg["order"] == "col" else (ri * nc + ci)
            vals = pd.to_numeric(df.iloc[cfg["r0"] + idx, 1:7], errors="coerce").tolist()
            pub = float(np.nanmean(vals))
            cell = panel[int(ri * ph / nr):int((ri + 1) * ph / nr),
                         int(ci * pw / nc):int((ci + 1) * pw / nc)]
            ch, cw = cell.shape[:2]
            ps = []
            for k in range(cfg["nrep"]):
                strip = cell[:, int(k * cw / cfg["nrep"]):int((k + 1) * cw / cfg["nrep"])]
                lp, sp, pc = color_severity(strip)
                if lp > 250:
                    ps.append(pc)
            pred = float(np.mean(ps)) if ps else 0.0
            recs.append(dict(sample_id="ara_%s_%s_%s" % (cfg["fig"], col, row.replace(":", "")),
                             crop="arabidopsis", batch=cfg["fig"], disease="gray mould",
                             group="%s-%s" % (col, row), published=pub, pred=pred,
                             pred_type="lesion_area_pct", n_rep=len(ps)))

# ---------- Rice and mango: processed single-leaf PNGs ----------
gt = pd.read_csv(LEAVES + "/ground_truth.csv", encoding="utf-8-sig")
for _, row in gt.iterrows():
    if row.crop not in ("rice", "mango"):
        continue
    p = LEAVES + "/" + row.sample_id + ".png"
    bgr = cv2.imread(p)
    lp, sp, pc = color_severity(bgr)
    recs.append(dict(sample_id=row.sample_id, crop=row.crop,
                     batch=("%s_Fig1" % row.crop) if row.crop == "rice" else "mango_S11",
                     disease=row.disease, group=row.group_name,
                     published=float(row.published_di_or_area), pred=pc,
                     pred_type="lesion_area_pct", n_rep=1))

# ---------- RoCoLe: official OIRSA grading, validation only ----------
coco = json.load(open(ROC + "/rocole-coco.json", encoding="utf-8"))
anns = {}
for a in coco["annotations"]:
    if a["category_id"] != 1:
        continue
    img = next(i for i in coco["images"] if i["id"] == a["image_id"])
    m = re.search(r"(C\d+P\d+[EH]\d+\.jpg)$", img["file_name"].rsplit("/", 1)[-1], re.I)
    if m:
        anns[m.group(1)] = (a, img.get("width", 2048), img.get("height", 1152))
man = pd.read_csv(ROC + "/rocole_subset_manifest.csv")
mid = {1: 3.0, 2: 13.0, 3: 35.5, 4: 75.0}  # OIRSA interval midpoints
for _, r in man.iterrows():
    fname = r["filename"]
    if fname not in anns:
        continue
    a, cw, ch = anns[fname]
    bgr = cv2.imread(ROC + "/" + fname)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    sx, sy = w / float(cw), h / float(ch)
    for poly in a["segmentation"]:
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        pts[:, 0] *= sx; pts[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, red = [bgr[..., i].astype(np.int16) for i in range(3)]
    fg = (mask > 0) & (hsv[..., 2] > 48) & (hsv[..., 1] > 35)
    lesion = fg & (red >= g * 0.94) & (g > 45) & (b < g * 0.92)
    pc = 100.0 * lesion.sum() / max(int(fg.sum()), 1)
    lvl = int(r["severity_level"])
    recs.append(dict(sample_id="rocole_%02d_%s" % (lvl, fname[:-4]), crop="rocole",
                     batch="RoCoLe_OIRSA", disease="coffee rust", group="rust_level_%d" % lvl,
                     published=mid[lvl], pred=pc, pred_type="lesion_area_pct", n_rep=1))

df = pd.DataFrame(recs)
df.to_csv(OUT + "/predictions_v2.csv", index=False)
print("total", len(df), df.groupby("crop").size().to_dict(), flush=True)
for name, sub in [("ALL", df),
                  ("continuous", df[df.crop != "rocole"]),
                  ("arabidopsis_Fig6_Fig7", df[df.batch.isin(["Fig6", "Fig7"])])]:
    sub = sub.copy()
    sub["z_pub"] = sub.groupby("batch").published.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    sub["z_pred"] = sub.groupby("batch").pred.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    print("%-24s n=%2d raw_r=%+.3f pooled_r=%.4f pooled_rho=%.4f" %
          (name, len(sub), sub.published.corr(sub.pred),
           sub.z_pub.corr(sub.z_pred), sub.z_pub.corr(sub.z_pred, method="spearman")), flush=True)
    if name == "ALL":
        df["z_pub"] = sub["z_pub"]; df["z_pred"] = sub["z_pred"]
        df.to_csv(OUT + "/predictions_v2.csv", index=False)
