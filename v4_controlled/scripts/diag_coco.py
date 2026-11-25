#!/usr/bin/env python3
import json, os, requests

ROOT = "."
META = os.path.join(ROOT, "data/raw_papers/mendeley_meta.json")
OUT = os.path.join(ROOT, "data/rocole_subset/rocole-coco.json")

meta = json.load(open(META))
print("candidates:", [repr(f["filename"]) for f in meta["files"] if "coco" in f["filename"].lower()], flush=True)
for f in meta["files"]:
    if "coco" in f["filename"].lower() and f["filename"].lower().endswith(".json"):
        url = f["content_details"]["download_url"]
        print("URL:", url[:160], flush=True)
        r = requests.get(url, timeout=240)
        print("HTTP", r.status_code, "bytes", len(r.content), flush=True)
        open(OUT, "wb").write(r.content)
        d = r.json()
        print("keys", list(d.keys()), flush=True)
        print("images", len(d.get("images", [])), "anns", len(d.get("annotations", [])), flush=True)
        print("cats", [(c["id"], c["name"]) for c in d.get("categories", [])], flush=True)
        print("img0", d["images"][0], flush=True)
        a = d["annotations"][0]
        print("ann0", {k: (v if k != "segmentation" else str(v)[:120]) for k, v in a.items()}, flush=True)
        break
else:
    print("NOT FOUND", flush=True)
