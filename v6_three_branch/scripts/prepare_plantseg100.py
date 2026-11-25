#!/usr/bin/env python3
"""V6 PlantSeg-100 sampling: v5's deterministic 48 + 52 more, same seed & strata.

- Test split ONLY (external validation, never tuned on).
- Seed 42, stratified by Plant, cap PER_PLANT images per plant, round-robin fill.
- v5 used PER_PLANT=2, N=48 -> these 48 are EXACTLY preserved (subset check).
- v6 uses PER_PLANT=4, N=100 with the SAME seed and same round-robin order, so
  the first 48 picks are unchanged and the extra 52 are drawn by the SAME rule.
  No image is filtered by mask ratio, prediction, or any outcome.
"""
import os, shutil
import numpy as np, pandas as pd, cv2

EXT = "./v5_formal/data/trackC_plantseg_gt/extracted/plantseg"
V5_META = "./v5_formal/data/trackC_meta.csv"
OUT = "./v6_three_branch/data/plantseg100"
SEED = 42
N_TARGET = 100
N_V5 = 48


def stratified_pick(test, per_plant, n_target, seed):
    rng = np.random.RandomState(seed)
    groups = {}
    for plant, g in test.groupby("Plant"):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        groups[plant] = idx[:per_plant].tolist()
    order = []
    rng2 = np.random.RandomState(seed + 1)
    for p in sorted(groups):
        rng2.shuffle(groups[p])
    round_i = 0
    while len(order) < n_target:
        added = False
        for p in sorted(groups):
            if round_i < len(groups[p]):
                order.append(groups[p][round_i]); added = True
                if len(order) >= n_target:
                    break
        if not added:
            break
        round_i += 1
    return sorted(order)


meta = pd.read_csv(EXT + "/Metadata.csv")
test = meta[meta.Split == "Test"].copy().reset_index(drop=True)
v5_meta = pd.read_csv(V5_META)
# v5 sample_id = "plantseg_" + Name.replace(".jpg","")
v5_names = set(v5_meta.sample_id.str.replace("plantseg_", "", regex=False) + ".jpg")
pick48 = test.index[test.Name.isin(v5_names)].tolist()
assert len(pick48) == 48, "must recover exactly the 48 v5 Test images, got %d" % len(pick48)

# Top up with 52 MORE, using the SAME seed & stratified-by-Plant rule, excluding v5's 48.
# No outcome-based filtering anywhere.
remaining = test.drop(index=pick48).reset_index(drop=True)
rng = np.random.RandomState(SEED)
groups = {}
for plant, g in remaining.groupby("Plant"):
    idx = g.index.values.copy()
    rng.shuffle(idx)
    groups[plant] = idx.tolist()
order = []
rng2 = np.random.RandomState(SEED + 1)
for p in sorted(groups):
    rng2.shuffle(groups[p])
round_i = 0
while len(order) < (N_TARGET - len(pick48)):
    added = False
    for p in sorted(groups):
        if round_i < len(groups[p]):
            order.append(groups[p][round_i]); added = True
            if len(order) >= (N_TARGET - len(pick48)):
                break
    if not added:
        break
    round_i += 1
# map remaining's positional indices back to original test index via Name
extra_names = [remaining.loc[j, "Name"] for j in order]
extra_idx = test.index[test.Name.isin(extra_names)].tolist()
pick100 = sorted(pick48 + extra_idx)
assert len(set(pick100)) == 100, "expected 100 unique images, got %d" % len(set(pick100))
print("Test split:", len(test), "plants:", test.Plant.nunique())
print("v5 48 preserved: True | newly added:", len(extra_idx))


def find(item, sub):
    p = EXT + "/" + sub + "/test/" + item
    if os.path.exists(p):
        return p
    import glob
    hits = glob.glob(EXT + "/" + sub + "/**/" + item, recursive=True)
    return hits[0] if hits else None


os.makedirs(OUT + "/imgs", exist_ok=True)
os.makedirs(OUT + "/masks", exist_ok=True)
rows = []
for i in pick100:
    r = test.loc[i]
    ip = find(r["Name"], "images")
    mp = find(r["Label file"], "annotations")
    if not ip or not mp:
        print("MISSING", r["Name"]); continue
    im = cv2.imread(ip)
    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if im is None or mask is None:
        print("BADREAD", r["Name"]); continue
    les = mask > 0
    frac = float(les.sum()) / float(mask.size)
    sid = "plantseg_" + r["Name"].replace(".jpg", "")
    cv2.imwrite(OUT + "/imgs/" + sid + ".jpg", im)
    cv2.imwrite(OUT + "/masks/" + sid + ".png", mask)
    in_v5 = sid in set(v5_meta.sample_id)
    rows.append(dict(sample_id=sid, track="C_plantseg", crop=str(r.Plant),
                     disease=str(r.Disease), unit_type="pixel_gt", published=frac,
                     metadata_mask_ratio=float(r["Mask ratio"]),
                     img_w=im.shape[1], img_h=im.shape[0],
                     value_source="expert_pixel_mask", split="Test",
                     in_v5_48=int(in_v5)))
df = pd.DataFrame(rows)
df.to_csv(OUT + "/meta100.csv", index=False)
err = (df.published - df.metadata_mask_ratio).abs()
print("Saved", len(df), "images; crops:", df.crop.nunique(), "diseases:", df.disease.nunique())
print("from v5-48:", int(df.in_v5_48.sum()), "| newly added:", int((df.in_v5_48 == 0).sum()))
print("max |computed - official Mask ratio| =", round(float(err.max()), 6))
print("published fraction stats:", df.published.describe().round(4).to_dict())
