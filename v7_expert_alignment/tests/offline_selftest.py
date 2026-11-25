#!/usr/bin/env python3
"""Network-free, data-free verification of every frozen v7 component.

Run: python v7_expert_alignment/tests/offline_selftest.py
This does NOT call DeepSeek and does NOT use any evaluation image.
"""
import json
import os
import sys

V7 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V7, "src"))
sys.path.insert(0, os.path.join(V7, "scripts"))

from metrics import selfcheck as metric_check          # noqa: E402
from llm_client_deepseek import selfcheck as llm_check  # noqa: E402
from polygon_and_clip import selfcheck as clip_check   # noqa: E402
from frozen_policy_v7 import policy_hash               # noqa: E402
from verify_manifest import main as verify             # noqa: E402


def main():
    out = {
        "metrics": metric_check(),
        "llm_client": llm_check(),
        "polygon_calibration": clip_check(),
        "policy_sha256": policy_hash(),
    }
    rc = verify()
    out["manifest_verification"] = "OK" if rc == 0 else "FAIL"
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
