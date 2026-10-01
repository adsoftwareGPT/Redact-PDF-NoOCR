#!/usr/bin/env python3
"""Standalone verification: compare a redacted PDF against its ORIGINAL.

Runs pixel-diff + LLM leak audit (+ optional auto-fix) without re-running
detection. Reads box coordinates from the redaction report JSON.

Usage:
  python tools/verify.py original.pdf redacted.pdf [--no-fix] [--workers 4]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redactor.config import DEFAULT_MODEL, get_api_key  # noqa: E402
from redactor.verify import run_verification  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original"); ap.add_argument("redacted")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    report_path = Path(args.redacted).with_suffix(".report.json")
    boxes = {}
    if report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        boxes = {p["page"]: [b["bbox_px"] for b in p["boxes"]] for p in rep["pages"]}
    v = run_verification(args.original, args.redacted, boxes, args.model, get_api_key(),
                         fix=not args.no_fix, workers=args.workers,
                         review_dir=str(Path(args.redacted).with_suffix("")) + "_review")
    print(json.dumps(v, ensure_ascii=False, indent=2)[:4000])
    print("\nRESULT:", "✅ OK" if v["ok"] else "⚠️ issues found — see above")


if __name__ == "__main__":
    main()
