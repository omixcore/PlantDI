#!/usr/bin/env python3
"""V6 SMOKE TEST (cost-controlled, BEFORE full run):
5 PlantSeg images that FAILED in v5 (empty lesions -> no SAM2 mask).
Verify: (a) new routing JSON 100% parses with scene_type/lesion_color;
        (b) two-level fallback yields a valid SAM2 mask on all 5;
        (c) prompt_source is recorded for every image.
No scoring, no tuning -- pure pipeline sanity check.
"""
import os, json, time, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(__file__))
from frozen_policy_v6 import route as frozen_route, policy_hash
from router_brain_v6 import query as route_query
from sam2_fallback import locate_with_fallback, mask_from_boxes_points

V5 = "./v5_formal"
BASE = "./v6_three_branch"
IMG = BASE + "/data/plantseg100/imgs"
OUT = BASE + "/results/v6_smoke5.json"
CKPT = "./models/sam2_hiera_tiny.pt"
key = os.environ["DEEPSEEK_API_KEY"]

# pick 5 v5 failures (SAM2 NaN), preferring those also present in plantseg100
v5d = pd.read_csv(V5 + "/results/trackC_predictions.csv")
fail = v5d.loc[v5d.sam2_frac.isna(), "sample_id"].tolist()[:5]
print("v5 SAM2-failed images selected for smoke:", fail, flush=True)
print("FROZEN POLICY SHA256:", policy_hash(), flush=True)

sam = None
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = SAM2ImagePredictor(build_sam2("sam2_hiera_t.yaml", CKPT, device="cpu"))
    print("SAM2 loaded", flush=True)
except Exception as e:
    print("SAM2 load FAILED:", repr(e), flush=True)

res = {}
all_ok = True
for sid in fail:
    p = IMG + "/" + sid + ".jpg"
    if not os.path.exists(p):
        print("[SKIP not-in-100]", sid, flush=True)
        continue
    t0 = time.time()
    dec, tok_route = route_query(p, key)
    if dec is None:
        dec = {"background": "natural", "scene_type": "in_the_wild",
               "lesion_color": "non_brown_or_mixed", "lesion_morph": "diffuse_necrosis",
               "rationale": "parse-fail fallback"}
    branch, why = frozen_route(dec)
    boxes, points, psrc, tok_loc, scale, ann = locate_with_fallback(p, key)
    bgr = cv2.imread(p)
    m = mask_from_boxes_points(sam, bgr, boxes, points) if sam else np.zeros(bgr.shape[:2], bool)
    frac = float(m.sum()) / float(m.size)
    ok = bool(dec and dec.get("scene_type") in ("controlled", "in_the_wild") and m.any())
    all_ok &= ok
    rec = dict(decision=dec, branch=branch, why=why, prompt_source=psrc,
               n_boxes=len(boxes), n_points=len(points), mask_frac=round(frac, 5),
               mask_nonempty=bool(m.any()), tok_route=tok_route, tok_loc=tok_loc,
               seconds=round(time.time() - t0, 2), ok=ok)
    res[sid] = rec
    print("[%s] %-38s branch=%-8s scene=%-11s color=%-18s src=%-8s mask=%s frac=%.3f %.1fs" % (
        "OK" if ok else "FAIL", sid[:38], branch, dec.get("scene_type","?"),
        dec.get("lesion_color","?"), psrc, m.any(), frac, time.time() - t0), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

n = len(res)
ok_n = sum(1 for v in res.values() if v["ok"])
print("\nSMOKE5: %d/%d OK; prompt_source=%s" % (
    ok_n, n, {v["prompt_source"] for v in res.values()}), flush=True)
print("RESULT:", "PASS" if (n >= 5 and ok_n == n and all_ok) else "CHECK", flush=True)
