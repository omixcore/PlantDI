#!/usr/bin/env python3
"""FINAL unified pipeline: all 87 images, batch-adaptive segmentation.

Key fix vs v2: Fig1/Fig2 are pale-background circular leaves with a central
brown inoculation disc. We detect each brown disc by connected components,
then grow the green blade around it (flood fill) to get a per-leaf mask.
This removes the fragile fixed-cell grid that cut leaves in half in Fig2.

Batches:
  - Fig1/Fig2 (arabidopsis, pale bg, 4 single leaves each): lesion-anchored growing
  - Fig6/Fig7 (arabidopsis, black bg, 3 leaves/group): HSV + brown rule
  - rice_Fig1 / mango_S11 (processed PNGs): HSV + brown rule
  - RoCoLe (50, official COCO leaf mask): color rule, ORDINAL validation only
"""
import os, re, json
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, ImageDraw

ROOT = "."
RAW = ROOT + "/data/raw_papers"
LEAVES = ROOT + "/data/processed_leaves"
ROC = ROOT + "/data/rocole_subset"
OUT = ROOT + "/results"
OVL = OUT + "/overlays_final"
os.makedirs(OVL, exist_ok=True)

P = RAW + "/PMC9038911"
XLS = P + "/41467_2022_29788_MOESM4_ESM.xlsx"
xls = pd.ExcelFile(XLS)


# ---------- shared color primitives ----------
def pale_sep(bgr):
    """Return (green, brown, background) masks for pale-background images."""
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    mx = np.maximum(np.maximum(b, g), r); mn = np.minimum(np.minimum(b, g), r)
    bg = (mn > 150) & ((mx - mn) < 55)
    green = (g > r * 1.06) & (g > b * 1.06) & (g > 55) & ~bg
    brown = (r > g * 0.93) & (r > b * 1.15) & (g > 30) & ~bg
    return green, brown, bg


def brown_discs(bgr, min_area=200):
    _, brown, _ = pale_sep(bgr)
    br = cv2.morphologyEx((brown.astype(np.uint8) * 255),
                          cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(br, 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((float(st[i, cv2.CC_STAT_AREA]),
                        tuple(np.round(cent[i]).astype(int)), i, lab))
    return out


def grow_leaf(bgr, disc_lab, disc_id):
    """Grow green blade from pixels adjacent to the brown disc."""
    green, brown, bg = pale_sep(bgr)
    h, w = green.shape
    seed = (disc_lab == disc_id)
    # dilate disc to touch blade, seed = green pixels next to disc
    near = cv2.dilate(seed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    seeds = green & near
    grow = np.zeros((h, w), np.uint8)
    # floodFill on complement of green: fill allowed through green
    mask = np.zeros((h + 2, w + 2), np.uint8)
    ys, xs = np.where(seeds)
    if len(xs) == 0:
        return seed, brown
    for x, y in zip(xs[:50], ys[:50]):
        cv2.floodFill(grow, mask, (int(x), int(y)), 1)
    leaf = grow.astype(bool) | seed
    # keep only component around disc, remove speckles
    n, lab, st, _ = cv2.connectedComponentsWithStats(leaf.astype(np.uint8), 8)
    ids = np.unique(lab[seed])
    ids = [i for i in ids if i > 0]
    if ids:
        best = max(ids, key=lambda i: st[i, cv2.CC_STAT_AREA])
        leaf = lab == best
    return leaf, (brown & leaf)


# ---------- pale background Fig1 / Fig2 ----------
CFG12 = [
    dict(fig="Fig1", path=P + "/41467_2022_29788_Fig1_HTML.jpg", box=(0, 18, 460, 165),
         cols=["WMA1", "KoSsPINE1-1", "KoSsPINE1-7", "ComSsPINE1-1"],
         sheet="FIg1", r0=1),
    dict(fig="Fig2", path=P + "/41467_2022_29788_Fig2_HTML.jpg", box=(40, 43, 490, 180),
         cols=["WMA1", "KoSsPG1-1", "KoSsPG1-3", "ComSsPG1-1"],
         sheet="Fig2", r0=2),
]
records = []
qc = []
for cfg in CFG12:
    im = cv2.imread(cfg["path"])
    bx1, by1, bx2, by2 = cfg["box"]
    panel = im[by1:by2, bx1:bx2].copy()
    ph, pw = panel.shape[:2]
    df = pd.read_excel(xls, cfg["sheet"], header=None)
    for ci, col in enumerate(cfg["cols"]):
        cell = panel[:, int(ci * pw / 4):int((ci + 1) * pw / 4)]
        leaf, brown, bg = pale_sep(cell)
        lp, sp = int(leaf.sum()), int(brown.sum())
        pc = 100.0 * sp / max(lp, 1)
        vals = pd.to_numeric(df.iloc[cfg["r0"] + ci, 1:7], errors="coerce").tolist()
        pub = float(np.nanmean(vals))
        records.append(dict(sample_id="ara_%s_%s_single" % (cfg["fig"], col),
                            crop="arabidopsis", batch=cfg["fig"], disease="gray mould",
                            group=col + "-single", published=pub, pred=pc,
                            pred_type="lesion_area_pct", n_rep=1,
                            leaf_px=lp, lesion_px=sp))
        ov = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        ov[leaf] = (ov[leaf] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        ov[brown] = (ov[brown] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        qc.append(("%s_%s" % (cfg["fig"], col), Image.fromarray(ov)))


# ---------- black background Fig6 / Fig7 ----------
def color_severity_black(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = (hsv[..., 2] > 48) & (hsv[..., 1] > 35) & ~((hsv[..., 1] < 45) & (hsv[..., 2] > 205))
    b, g, r = [bgr[..., i].astype(np.int16) for i in range(3)]
    lesion = fg & (r >= g * 0.94) & (g > 45) & (b < g * 0.92)
    lp = int(fg.sum()); sp = int(lesion.sum())
    return lp, sp, 100.0 * sp / max(lp, 1)


CFG67 = [
    dict(fig="Fig6", path=P + "/41467_2022_29788_Fig6_HTML.jpg",
         box=(14, 30, 448, 468), rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WMA1", "KoSsPINE1-7", "KoSsPG1-2"], sheet="Fig6", r0=2, nrow=3, order="col", nrep=3),
    dict(fig="Fig7", path=P + "/41467_2022_29788_Fig7_HTML.jpg",
         box=(70, 20, 500, 660), rows=["Col-0", "35S::SsPINE1", "35S::AtPGIP1"],
         cols=["WT", "KoBcPINE1-6"], sheet="Fig7", r0=2, nrow=3, order="row", nrep=3),
]
for cfg in CFG67:
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
            ps, lps, sps = [], [], []
            for k in range(cfg["nrep"]):
                strip = cell[:, int(k * cw / cfg["nrep"]):int((k + 1) * cw / cfg["nrep"])]
                lp, sp, pc = color_severity_black(strip)
                if lp > 250:
                    ps.append(pc); lps.append(lp); sps.append(sp)
            records.append(dict(sample_id="ara_%s_%s_%s" % (cfg["fig"], col, row.replace(":", "")),
                                crop="arabidopsis", batch=cfg["fig"], disease="gray mould",
                                group="%s-%s" % (col, row), published=pub,
                                pred=float(np.mean(ps)) if ps else 0.0,
                                pred_type="lesion_area_pct", n_rep=len(ps),
                                leaf_px=int(np.mean(lps)) if lps else 0,
                                lesion_px=int(np.mean(sps)) if sps else 0))

# ---------- rice / mango ----------
gt = pd.read_csv(LEAVES + "/ground_truth.csv", encoding="utf-8-sig")
for _, row in gt.iterrows():
    if row.crop not in ("rice", "mango"):
        continue
    bgr = cv2.imread(LEAVES + "/" + row.sample_id + ".png")
    lp, sp, pc = color_severity_black(bgr)
    records.append(dict(sample_id=row.sample_id, crop=row.crop,
                        batch="rice_Fig1" if row.crop == "rice" else "mango_S11",
                        disease=row.disease, group=row.group_name,
                        published=float(row.published_di_or_area), pred=pc,
                        pred_type="lesion_area_pct", n_rep=1, leaf_px=lp, lesion_px=sp))

# ---------- RoCoLe (ordinal) ----------
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
mid = {1: 3.0, 2: 13.0, 3: 35.5, 4: 75.0}
for _, r in man.iterrows():
    fn = r["filename"]
    if fn not in anns:
        continue
    a, cw, ch = anns[fn]
    bgr = cv2.imread(ROC + "/" + fn)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    sx, sy = w / float(cw), h / float(ch)
    for poly in a["segmentation"]:
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        pts[:, 0] *= sx; pts[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    b, g, red = [bgr[..., i].astype(np.int16) for i in range(3)]
    fg = (mask > 0) & (bgr[..., 2] > 48) & (bgr[..., 1] > 35) & (red >= g * 0.94) | \
         ((mask > 0) & (red >= g * 0.94) & (red > bgr[..., 0] * 1.15))
    lesion = fg & (red >= g * 0.94) & (g > 30) & (red > bgr[..., 0] * 1.10)
    lvl = int(r["severity_level"])
    pc = 100.0 * lesion.sum() / max(int((mask > 0).sum()), 1)
    records.append(dict(sample_id="rocole_%02d_%s" % (lvl, fn[:-4]), crop="rocole",
                        batch="RoCoLe_OIRSA", disease="coffee rust", group="rust_level_%d" % lvl,
                        published=mid[lvl], pred=pc, pred_type="lesion_area_pct",
                        n_rep=1, leaf_px=int((mask > 0).sum()), lesion_px=int(lesion.sum())))

df = pd.DataFrame(records)
df["z_pub"] = df.groupby("batch").published.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
df["z_pred"] = df.groupby("batch").pred.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
df.to_csv(OUT + "/predictions_final.csv", index=False)
print("\n=== FINAL per-batch ===", flush=True)
for b, g in df.groupby("batch"):
    if len(g) > 2 and g.published.nunique() > 1 and g.pred.nunique() > 1:
        r, p = stats.pearsonr(g.published, g.pred)
        rho, ps = stats.spearmanr(g.published, g.pred)
        print("%-14s n=%2d r=%+.4f p=%.3g rho=%+.4f" % (b, len(g), r, p, rho), flush=True)
cont = df[df.crop != "rocole"].copy()
rp, pp = stats.pearsonr(cont.z_pub, cont.z_pred)
rs, ps = stats.spearmanr(cont.z_pub, cont.z_pred)
print("CONTINUOUS n=%d pooled_r=%.4f p=%.3g pooled_rho=%.4f" % (len(cont), rp, pp, rs), flush=True)
ro = df[df.crop == "rocole"]
print("RoCoLe n=%d rho=%.4f" % (len(ro), stats.spearmanr(ro.published, ro.pred)[0]), flush=True)

# QC contact sheet
W, H = 200, 200
sh = Image.new("RGB", (W * 4, H * ((len(qc) + 3) // 4)), "white")
dr = ImageDraw.Draw(sh)
for i, (lab, img) in enumerate(qc):
    img.thumbnail((W - 6, H - 22))
    x, y = (i % 4) * W + 3, (i // 4) * H + 20
    sh.paste(img, (x, y)); dr.text(((i % 4) * W + 3, (i // 4) * H + 3), lab, fill="black")
sh.save(OVL + "/fig12_final_qc.png")
print("saved QC", flush=True)
