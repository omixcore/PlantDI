#!/usr/bin/env python3
"""V6 three-branch routing brain. MLLM zero-shot, IMAGE PIXELS ONLY.

Adds two fields vs v4: scene_type (controlled|in_the_wild), lesion_color
(brown_necrosis|non_brown_or_mixed). No filename / crop / batch / disease name
is ever passed to the model -- only the downscaled image bytes.

Writes a JSON cache keyed by sample_id. Reuses frozen_policy_v6.route() for the
final branch decision so the brain itself never chooses a branch directly.
"""
import os, base64, json, time, argparse
import cv2, requests

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

_LINES = [
 "You are a computer-vision routing controller for plant disease phenotyping.",
 "Look ONLY at the image. Output ONLY one strict JSON object, nothing else:",
 "{",
 '  "background": "pale" | "dark" | "natural",',
 '  "scene_type": "controlled" | "in_the_wild",',
 '  "lesion_color": "brown_necrosis" | "non_brown_or_mixed",',
 '  "lesion_morph": "isolated_disc" | "diffuse_necrosis" | "scattered_flecks",',
 '  "rationale": "<one short sentence>"',
 "}",
 "Definitions (decide from the image alone):",
 '- background: pale = near-white uniform backdrop (lab scanner / detached-leaf panel);',
 '  dark = black/dark uniform backdrop; natural = field / soil / greenhouse background.',
 '- scene_type controlled = a single detached leaf, a tidy lab panel, or a controlled',
 '  inoculation photo with a uniform artificial background, typical of a paper figure.',
 '  scene_type in_the_wild = natural scene: soil, canopy, multiple overlapping leaves,',
 '  uneven illumination, field or greenhouse photography.',
 '- lesion_color brown_necrosis = the lesions are clearly brown / dark-brown / black',
 '  necrotic spots contrasting with green healthy tissue.',
 '  lesion_color non_brown_or_mixed = lesions are greenish, yellow (chlorosis), rust-orange,',
 '  purple, water-soaked, mosaic, fleck, fruit rot, or any non-brown / mixed hue.',
 '- lesion_morph isolated_disc = ONE sharply bounded circular/oval spot on an otherwise',
 '  healthy blade; diffuse_necrosis = irregular spreading browning/yellowing;',
 '  scattered_flecks = many small scattered spots across the foliage.',
 "Return ONLY the JSON object, starting with an opening brace and ending with a closing brace.",
]
SYSTEM = "\n".join(_LINES)
USER = "Inspect this leaf image and output the routing JSON."


def parse(t):
    if not t:
        return None
    i = t.find("{")
    if i < 0:
        return None
    dep = 0
    in_str, esc = False, False
    for j in range(i, len(t)):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            dep += 1
        elif ch == "}":
            dep -= 1
            if dep == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    return None
    return None


def _encode(path, max_side=900, quality=88):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def query(path, key, max_retry=8):
    b64 = _encode(path)
    if b64 is None:
        return None, 0
    payload = {
        "model": MODEL, "effort": "low", "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": USER},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
        ],
    }
    hdr = {"Authorization": "Bearer " + key}
    for a in range(max_retry):
        try:
            r = requests.post(API, headers=hdr, json=payload, timeout=90)
            d = r.json()
            if r.status_code != 200:
                time.sleep(1 + a)
                continue
            m = d["choices"][0]["message"]
            ann = parse(m.get("content") or "") or parse(m.get("reasoning_content") or "")
            if ann and ann.get("background") in ("pale", "dark", "natural"):
                ann.setdefault("scene_type", "in_the_wild" if ann["background"] == "natural" else "controlled")
                ann.setdefault("lesion_color", "brown_necrosis")
                ann.setdefault("lesion_morph", "diffuse_necrosis")
                ann.setdefault("rationale", "")
                if ann["scene_type"] not in ("controlled", "in_the_wild"):
                    ann["scene_type"] = "in_the_wild" if ann["background"] == "natural" else "controlled"
                if ann["lesion_color"] not in ("brown_necrosis", "non_brown_or_mixed"):
                    ann["lesion_color"] = "brown_necrosis"
                return ann, d.get("usage", {}).get("total_tokens", 0)
            time.sleep(1)
        except Exception:
            time.sleep(1 + a)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="CSV with sample_id column")
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True, help="JSON cache path")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"]
    import pandas as pd
    meta = pd.read_csv(args.meta)
    cache = json.load(open(args.out)) if os.path.exists(args.out) else {}
    rows = meta if not args.limit else meta.head(args.limit)
    for i, sid in enumerate(rows.sample_id.tolist(), 1):
        if sid in cache:
            print("[skip]", sid, flush=True)
            continue
        ann, tok = query(args.imgdir + "/" + sid + (".png" if os.path.exists(args.imgdir + "/" + sid + ".png") else ".jpg"), key)
        if ann:
            ann["_tokens"] = tok
            cache[sid] = ann
            json.dump(cache, open(args.out, "w"), indent=1)
            print("[%d/%d] %-34s scene=%-12s color=%-18s bg=%-7s morph=%s" % (
                i, len(rows), sid[:34], ann["scene_type"], ann["lesion_color"],
                ann["background"], ann["lesion_morph"]), flush=True)
        else:
            print("[ERR]", sid, flush=True)
    print("cached", len(cache), "->", args.out, flush=True)


if __name__ == "__main__":
    main()
