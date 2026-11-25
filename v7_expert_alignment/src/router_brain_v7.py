#!/usr/bin/env python3
"""T2 - v7 routing brain. Strict `controlled` boundary, ZERO SAM2 by default.

Only image pixels are sent to the MLLM (no filename / crop / batch / disease
name). The final branch is always decided by the frozen `frozen_policy_v7.route`,
never by the model directly. Parse failures are routed conservatively to the
VLM polygon branch.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from llm_client_deepseek import CallStats, MLLMReply as GeminiReply, chat_json, get_api_key  # noqa: E402
from prompts_v7 import (FROZEN_CONFIG_V7, ROUTE_SYSTEM_V7, ROUTE_USER_V7)  # noqa: E402
from frozen_policy_v7 import route as frozen_route  # noqa: E402


def _normalize(obj: Dict[str, Any] | None, parse_ok: bool) -> Dict[str, Any]:
    """Map model output into the frozen policy's controlled vocabulary."""
    obj = obj or {}
    bg_raw = str(obj.get("background", "")).strip().lower()
    if bg_raw in ("scanner_white", "white", "pale"):
        bg = "scanner_white"
    elif bg_raw in ("black_velvet", "black", "dark"):
        bg = "black_velvet"
    else:
        bg = "other"

    scene_raw = str(obj.get("scene_type", "")).strip().lower()
    if scene_raw in ("controlled", "lab", "laboratory", "scanner"):
        scene = "controlled"
    elif scene_raw in ("in_the_wild", "wild", "natural", "field"):
        scene = "in_the_wild"
    else:
        scene = "in_the_wild"

    color_raw = str(obj.get("lesion_color", "")).strip().lower()
    if color_raw in ("brown_necrosis", "brown", "necrosis"):
        color = "brown_necrosis"
    else:
        color = "non_brown_or_mixed"

    morph_raw = str(obj.get("lesion_morph", "")).strip().lower()
    morph = morph_raw if morph_raw in ("isolated_disc", "diffuse_necrosis", "scattered_flecks") else ""

    return {
        "background": bg,
        "scene_type": scene,
        "lesion_color": color,
        "lesion_morph": morph,
        "rationale": str(obj.get("rationale", ""))[:200],
        "prompt_source": "primary" if parse_ok else "parse_fail_conservative",
    }


def query(image_path: str, *, model: str = "deepseek-flash") -> Tuple[Dict[str, Any], CallStats]:
    """Return (normalized decision, call statistics). Key comes from the env only."""
    key = get_api_key()
    reply: GeminiReply = chat_json(
        ROUTE_SYSTEM_V7, ROUTE_USER_V7, image_path,
        model=model,
        max_output_tokens=FROZEN_CONFIG_V7["t1_max_output_tokens"],
        reasoning_effort=FROZEN_CONFIG_V7["t1_reasoning_effort"],
        key=key,
    )
    dec = _normalize(reply.parsed, reply.parsed is not None)
    branch, why = frozen_route(dec)
    dec["branch"] = branch
    dec["branch_reason"] = why
    return dec, reply.stats


def decide(dec: Dict[str, Any]) -> Tuple[str, str]:
    """Freeze-policy wrapper for cached decisions."""
    return frozen_route(dec)


def main() -> None:
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="CSV with sample_id column")
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True, help="JSON cache path")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    cache: Dict[str, Any] = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    rows = meta if not args.limit else meta.head(args.limit)
    for i, sid in enumerate(rows.sample_id.tolist(), 1):
        if sid in cache:
            print("[skip]", sid, flush=True)
            continue
        p = os.path.join(args.imgdir, str(sid) + ".jpg")
        if not os.path.exists(p):
            p = os.path.join(args.imgdir, str(sid) + ".png")
        try:
            dec, st = query(p, model=args.model)
        except Exception as exc:
            print("[ERR]", sid, repr(exc), flush=True)
            continue
        cache[sid] = {
            **dec,
            "_tokens_in": st.input_tokens,
            "_tokens_out": st.output_tokens,
            "_seconds": st.latency_s,
            "_model": st.model,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, ensure_ascii=False)
        print("[%d/%d] %-36s scene=%-12s bg=%-14s branch=%s  %.1fs" % (
            i, len(rows), str(sid)[:36], dec["scene_type"], dec["background"],
            dec["branch"], st.latency_s), flush=True)
    print("cached", len(cache), "->", args.out, flush=True)


if __name__ == "__main__":
    main()
