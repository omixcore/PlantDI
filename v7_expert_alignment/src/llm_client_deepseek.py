#!/usr/bin/env python3
"""T1 - DeepSeek MLLM client (OpenAI-compatible), CPU-only friendly.

Backend switched from Gemini to DeepSeek on user request 2026-11-24, BEFORE any
test-set evaluation. Endpoint and model were verified live with a real leaf
image: `deepseek-flash`, `effort=high`, base64 original image, token accounting.

Key rules (audit):
  * Key is read ONLY from os.environ["DEEPSEEK_API_KEY"]; never written to any
    file, log, JSON cache, exception message or result CSV.
  * NO downscaling: the exact stored image bytes are base64-encoded.
  * Every call records wall-clock seconds and input/output token counts.
  * The `effort` field (DeepSeek reasoning control) is sent and its acceptance
    is recorded per call. A 400 that names the field is retried once without it.

Endpoint: https://api.deepseek.com/chat/completions
Model:    deepseek-flash (default), overridable via FROZEN_CONFIG_V7 / --model.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

BASE_URL = "https://api.deepseek.com"
CHAT_URL = BASE_URL + "/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
SUPPORTED_MODELS = ("deepseek-flash", "deepseek-chat", "deepseek-reasoner")


class MissingAPIKey(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(
            "DEEPSEEK_API_KEY is not set. Export it in the shell before running; "
            "never place the key in code, config files, or command lines."
        )
    return key


def key_fingerprint() -> str:
    """Non-reversible 8-hex fingerprint, safe to log for run identity checks."""
    try:
        import hashlib
        return hashlib.sha256(get_api_key().encode()).hexdigest()[:8]
    except MissingAPIKey:
        return "unset"


@dataclass
class CallStats:
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_effort_sent: str
    reasoning_payload_accepted: bool
    http_status: int
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MLLMReply:
    text: str
    parsed: Optional[Dict[str, Any]]
    stats: CallStats


# Backward-compatible alias so router/polygon modules need no functional change.
GeminiReply = MLLMReply


def _encode_original(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise ValueError("empty image file: " + path)
    return base64.b64encode(raw).decode("ascii")


def _mime_for(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp"}.get(ext, "image/jpeg")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from a model reply."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _usage_of(payload: Dict[str, Any]) -> tuple[int, int, int]:
    u = payload.get("usage") or {}
    if not isinstance(u, dict):
        u = {}
    inp = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    out = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    total = int(u.get("total_tokens") or (inp + out))
    return inp, out, total


def _safe_err(exc: BaseException, key: str) -> str:
    msg = repr(exc)
    return msg.replace(key, "<redacted>") if key else msg


def chat_json(
    system_prompt: str,
    user_text: str,
    image_path: Optional[str] = None,
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 4096,
    reasoning_effort: str = "high",
    temperature: float = 0.0,
    timeout: int = 180,
    max_retries: int = 3,
    key: Optional[str] = None,
) -> MLLMReply:
    """Call DeepSeek and return parsed JSON plus full call statistics.

    Retries only on transport / 429 / 5xx. Parse failure is returned with
    parsed=None so the caller can trigger the frozen primary->retry->fallback
    chain without blind re-calling.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError("unsupported model: %s" % model)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for DeepSeek calls") from exc
    key = key or get_api_key()
    b64 = _encode_original(image_path) if image_path else None
    user_content: Any = user_text
    if b64 is not None:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (_mime_for(image_path), b64),
            }},
        ]

    base_payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last_status = 0
    last_error = ""
    t0 = time.perf_counter()

    for attempt in range(max_retries + 1):
        payload = dict(base_payload)
        effort_sent = bool(reasoning_effort)
        if reasoning_effort:
            payload["effort"] = reasoning_effort
        try:
            resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=timeout)
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                # DeepSeek may put reasoning in a separate field; merge for parsing.
                text = (msg.get("content") or "")
                reasoning = msg.get("reasoning_content") or ""
                combined = (text + "\n" + reasoning).strip() if reasoning else text
                inp, out, total = _usage_of(data)
                stats = CallStats(model, round(time.perf_counter() - t0, 3), inp, out, total,
                                  reasoning_effort or "", effort_sent, 200)
                return MLLMReply(text=combined, parsed=extract_json(combined), stats=stats)
            body = resp.text[:600].replace(key, "<redacted>")
            last_error = "HTTP %d: %s" % (resp.status_code, body)
            if resp.status_code == 400 and reasoning_effort and "effort" in body.lower():
                reasoning_effort = ""  # retry once without the field
                continue
            if resp.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_error = _safe_err(exc, key)
        time.sleep(min(2 ** attempt, 8))

    stats = CallStats(model, round(time.perf_counter() - t0, 3), 0, 0, 0,
                      reasoning_effort or "", False, last_status, last_error[:300])
    return MLLMReply(text="", parsed=None, stats=stats)


def selfcheck() -> Dict[str, Any]:
    """Network-free sanity check of encoding, JSON extraction and key sourcing."""
    sample = 'prefix {"a": 1, "b": [1,2]} suffix'
    assert extract_json(sample) == {"a": 1, "b": [1, 2]}
    assert extract_json("no json") is None
    return {
        "base_url": BASE_URL,
        "models": list(SUPPORTED_MODELS),
        "key_source": "os.environ['DEEPSEEK_API_KEY']",
        "key_fingerprint": key_fingerprint(),
        "downsample": "none (original bytes)",
        "json_parser": "ok",
    }


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2))
