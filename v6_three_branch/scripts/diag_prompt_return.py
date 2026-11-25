#!/usr/bin/env python3
"""Diagnose WHY primary/retry return no lesions: dump raw model output.
On 3 PlantSeg images, print content + reasoning_content for both prompts."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from sam2_fallback import _encode_small, _call, PRIMARY_SYSTEM, RETRY_SYSTEM, parse

BASE = "./v6_three_branch"
IMG = BASE + "/data/plantseg100/imgs"
key = os.environ["DEEPSEEK_API_KEY"]
ids = ["plantseg_apple_rust_4", "plantseg_bean_rust_16", "plantseg_tomato_early_blight_google_0001"]
import glob
avail = set(os.path.basename(p)[:-4] for p in glob.glob(IMG + "/*.jpg"))
ids = [s for s in ids if s in avail]
if len(ids) < 3:
    ids = sorted(avail)[:3]

for sid in ids:
    b64, scale, shape = _encode_small(IMG + "/" + sid + ".jpg")
    print("\n" + "=" * 70 + "\nIMAGE:", sid, "scale=", scale, flush=True)
    for name, sys_prompt in [("PRIMARY", PRIMARY_SYSTEM), ("RETRY", RETRY_SYSTEM)]:
        # call raw to capture both content and reasoning
        import requests, json, time
        payload = {"model": "deepseek-flash", "effort": "low", "max_tokens": 1200,
          "messages": [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": "JSON only:"},
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
        try:
            r = requests.post("https://api.deepseek.com/chat/completions",
                              headers={"Authorization": "Bearer " + key}, json=payload, timeout=90)
            d = r.json()
            m = d["choices"][0]["message"]
            c = (m.get("content") or "").strip()
            rc = (m.get("reasoning_content") or "").strip()
            print("[%s] status=%d content_len=%d reasoning_len=%d" % (name, r.status_code, len(c), len(rc)))
            print("  content[:300]:", repr(c[:300]))
            print("  parsed:", parse(c) or parse(rc))
        except Exception as e:
            print("[%s] ERR %r" % (name, e))
