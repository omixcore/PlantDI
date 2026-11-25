#!/usr/bin/env python3
import pandas as pd, os, glob
R = "./data/raw_papers"
for name, pat in [
    ("case1", R + "/PMC11039461/sourcedata/Source data/Source data.xlsx"),
    ("case2", R + "/PMC9038911/41467_2022_29788_MOESM4_ESM.xlsx"),
]:
    print("=" * 30, name, os.path.exists(pat))
    if not os.path.exists(pat):
        continue
    x = pd.ExcelFile(pat)
    print("sheets", len(x.sheet_names), x.sheet_names[:30])
    for s in x.sheet_names[:20]:
        df = pd.read_excel(x, s, header=None)
        txt = " ".join(str(v) for v in df.head(8).values.ravel()[:60] if str(v) != "nan")
        if any(k in txt.lower() for k in ["disease index", "lesion area", "lesion size", "%"]):
            print("--", s, df.shape)
            print(df.head(10).to_string(max_colwidth=45))
            print()
print("=== figure images ===")
for p in sorted(glob.glob(R + "/*/Fig*_HTML.jpg")):
    from PIL import Image
    im = Image.open(p)
    print(os.path.basename(p), im.size)
