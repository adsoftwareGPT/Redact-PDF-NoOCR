"""OpenRouter vision-LLM client with robust JSON extraction and retries."""
import json
import re
import sys
import time

import threading

import requests

from .config import API_URL

_SESSION = requests.Session()  # shared: reuses TLS connections across threads

_USAGE_LOCK = threading.Lock()
USAGE_TOTAL = {"calls": 0, "prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "cost_known": False}


def usage_snapshot() -> dict:
    """Copy of the cumulative usage meter (all calls via this process)."""
    with _USAGE_LOCK:
        return dict(USAGE_TOTAL)


def usage_diff(later: dict, earlier: dict) -> dict:
    return {k: round(later[k] - earlier[k], 6) for k in ("calls", "prompt", "completion", "total", "cost")} | {
        "cost_known": later["cost_known"]}


def _add_usage(u) -> None:
    if not isinstance(u, dict):
        return
    with _USAGE_LOCK:
        USAGE_TOTAL["calls"] += 1
        USAGE_TOTAL["prompt"] += int(u.get("prompt_tokens") or 0)
        USAGE_TOTAL["completion"] += int(u.get("completion_tokens") or 0)
        USAGE_TOTAL["total"] += int(u.get("total_tokens") or 0)
        c = u.get("cost")
        if c is not None:
            USAGE_TOTAL["cost"] += float(c)
            USAGE_TOTAL["cost_known"] = True

RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504, 529)
ATTEMPTS = 3


def call_vision_model(image_b64: str, prompt: str, model: str, api_key: str,
                      mime: str = "image/png", max_tokens: int = 12000,
                      temperature: float = 0.0, quiet: bool = False) -> str:
    """Send one image + prompt to the vision model, return raw text content."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,  # continuous bytes -> read timeout applies per chunk;
                         # long generations can't false-trigger it
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in API_URL:  # attribution headers, OpenRouter only
        headers["HTTP-Referer"] = "https://github.com/adsoftwareGPT/Redact-PDF-NoOCR"
        headers["X-Title"] = "Redact-PDF-NoOCR"
    last_err = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            resp = _SESSION.post(API_URL, headers=headers, json=payload,
                                 timeout=(30, 75), stream=True)
            if resp.status_code in RETRYABLE_STATUS and attempt < ATTEMPTS:
                wait = 2 ** attempt * 2
                if not quiet:
                    print(f"  ⏳ HTTP {resp.status_code}, retry {attempt}/{ATTEMPTS-1} in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            # consume the SSE stream, concatenating content deltas
            parts = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    _add_usage(obj["usage"])
                try:
                    delta = obj["choices"][0].get("delta", {})
                except (KeyError, IndexError):
                    continue
                parts.append(delta.get("content") or "")
            return "".join(parts)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < ATTEMPTS:
                wait = 2 ** attempt * 2
                if not quiet:
                    print(f"  ⏳ stalled ({type(e).__name__}), retry {attempt}/{ATTEMPTS-1} in {wait}s", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"vision model unreachable after {ATTEMPTS} attempts: {last_err}")


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"model returned no parsable JSON: {text[:200]!r}")
    body = text[start:]
    # repair truncated output (max_tokens hit mid-array): trim to the last
    # complete object, then try closing the open structures
    for _ in range(6):
        for suffix in ("]}", "]}", '"}]}'):
            try:
                return json.loads(body + suffix)
            except json.JSONDecodeError:
                continue
        cut = body.rfind("}")
        if cut <= 0:
            break
        body = body[:cut + 1]
    raise ValueError(f"model returned no parsable JSON: {text[:200]!r}")


def validate_boxes(raw_boxes: list) -> list:
    """Keep only well-formed boxes; clamp to 0-1000, ensure x2>x1, y2>y1."""
    out = []
    for b in raw_boxes if isinstance(raw_boxes, list) else []:
        try:
            bbox = b["bbox_2d"]
            x1, y1, x2, y2 = [max(0, min(1000, float(v))) for v in bbox[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({
                "bbox_2d": [x1, y1, x2, y2],
                "category": str(b.get("category", "other"))[:40],
                "text": str(b.get("text", ""))[:200],
            })
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out
