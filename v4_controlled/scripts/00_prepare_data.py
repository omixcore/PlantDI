#!/usr/bin/env python3
"""Prepare processed leaf images and ground_truth.csv for PlantDI MVP.

All published values are taken from the papers' own Source Data Excel files,
except case 3 (mango), where the values are digitized from Figure S11 bar chart
and explicitly flagged as figure-digitized.
"""
import os
import shutil
import pandas as pd
from PIL import Image

ROOT = os.environ.get("PLANTDI_ROOT", ".")
RAW = os.path.join(ROOT, "data", "raw_papers")
ROC = os.path.join(ROOT, "data", "rocole_subset")
OUT = os.path.join(ROOT, "data", "processed_leaves")
os.makedirs(OUT, exist_ok=True)

rows = []

# ---------- RoCoLe: ordinal rust severity level as published truth ----------
man = pd.read_csv(os.path.join(ROC, "rocole_subset_manifest.csv"))
for _, r in man.iterrows():
    sid = r["sample_id"]
    dst = os.path.join(OUT, sid + ".png")
    src = os.path.join(ROC, r["filename"])
    with Image.open(src) as im:
        im.convert("RGB").save(dst)
    rows.append(dict(
        sample_id=sid, crop="rocole", disease="coffee rust (Hemileia vastatrix)",
        group_name="rust_level_%d" % r["severity_level"],
        published_di_or_area=float(r["severity_level"]),
        grading_standard="RoCoLe ordinal rust_level 1-4 (no continuous area in dataset)",
        source_file=os.path.relpath(src, ROOT),
        value_source="dataset_classification_xlsx",
    ))

# ---------- Case 1: rice blast, Nat Commun 2024, PMC11039461 ----------
# Disease index (%) from "Source data.xlsx", sheet "Figure 1", panel Figure 1l.
# Panel letters a-j of Figure 1 are the corresponding leaf photographs.
di1 = {
    "a": ("LTH", 60.0), "b": ("IRBLa-A", 30.0), "c": ("IRBL19-A", 13.33333),
    "d": ("IRBLta-cp1", 20.0), "e": ("IRBL20-IR24", 26.66667),
    "f": ("IRBLi-F5", 33.33333), "g": ("IRBL3-CP4", 33.33333),
    "h": ("IRBLk-K", 20.0), "i": ("IRBLsh-S", 26.66667),
    "j": ("IRBLKs-S", 13.33333),
}
f1 = os.path.join(RAW, "PMC11039461", "41467_2024_47746_Fig1_HTML.jpg")
with Image.open(f1) as im:
    xs = [0, 149, 298, 447, 596, 746]
    ys = [0, 190, 380]
    for i, (panel, (grp, val)) in enumerate(di1.items()):
        rr, cc = divmod(i, 5)
        box = (xs[cc] + 22, ys[rr] + 8, xs[cc + 1] - 4, ys[rr] + 172)
        sid = "case1_fig1_%s" % panel
        im.crop(box).convert("RGB").save(os.path.join(OUT, sid + ".png"))
        rows.append(dict(
            sample_id=sid, crop="rice", disease="rice blast (Magnaporthe oryzae)",
            group_name=grp, published_di_or_area=val,
            grading_standard="Disease index (%) as reported in Source data.xlsx, Figure 1l",
            source_file=os.path.relpath(f1, ROOT),
            value_source="source_data_xlsx:PMC11039461/Figure 1",
        ))

# ---------- Case 2: Arabidopsis gray mould, Nat Commun 2022, PMC9038911 ----------
# Lesion area (cm2) means from MOESM4_ESM.xlsx, sheet "FIg1", Figure 1a.
les2 = [
    ("WMA1", 5.100, (5, 35, 105, 162)),
    ("KoSsPINE1-1", 1.727, (110, 35, 225, 162)),
    ("KoSsPINE1-7", 1.784, (230, 35, 340, 162)),
    ("ComSsPINE1-1", 5.233, (345, 35, 450, 162)),
]
f2 = os.path.join(RAW, "PMC9038911", "41467_2022_29788_Fig1_HTML.jpg")
with Image.open(f2) as im:
    for grp, val, box in les2:
        sid = "case2_fig1_" + grp.replace("-", "_").lower()
        im.crop(box).convert("RGB").save(os.path.join(OUT, sid + ".png"))
        rows.append(dict(
            sample_id=sid, crop="arabidopsis",
            disease="gray mould (Botrytis cinerea)",
            group_name=grp, published_di_or_area=val,
            grading_standard="Mean lesion area (cm2) of 3 replicates, Source Data Fig 1a",
            source_file=os.path.relpath(f2, ROOT),
            value_source="source_data_xlsx:PMC9038911/FIg1",
        ))

# ---------- Case 3: mango anthracnose, Mol Plant Pathol 2025, PMC12571544 ----------
# Figure S11 bar chart digitized at pixel level; y-axis ticks 20/25/30 -> values below.
les3 = [
    ("35S::Cglac8", 24.9, (465, 8, 640, 180)),
    ("35S::Cglac8+MiLRR-RLP1", 25.0, (465, 190, 640, 360)),
    ("35S::Cglac8+Mi14-3-3-D1", 23.2, (465, 370, 640, 540)),
    ("35S::Cglac8+MiLRR-RLP1+Mi14-3-3-D1", 21.1, (465, 550, 640, 730)),
]
f3 = os.path.join(RAW, "PMC12571544", "MPP-26-e70163-s013.jpg")
with Image.open(f3) as im:
    for k, (grp, val, box) in enumerate(les3, 1):
        sid = "case3_s11_%d" % k
        im.crop(box).convert("RGB").save(os.path.join(OUT, sid + ".png"))
        rows.append(dict(
            sample_id=sid, crop="mango",
            disease="anthracnose (Colletotrichum gloeosporioides)",
            group_name=grp, published_di_or_area=val,
            grading_standard="Lesion area (%) digitized from Figure S11 bar chart (y-axis 20-30%)",
            source_file=os.path.relpath(f3, ROOT),
            value_source="figure-digitized:PMC12571544/Figure S11",
        ))

gt = pd.DataFrame(rows, columns=[
    "sample_id", "crop", "disease", "group_name", "published_di_or_area",
    "grading_standard", "source_file", "value_source",
])
gt.to_csv(os.path.join(OUT, "ground_truth.csv"), index=False, encoding="utf-8-sig")
print("ground_truth rows:", len(gt))
print(gt.groupby("crop").size().to_string())
