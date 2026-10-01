#!/usr/bin/env python3
"""no-ocr-redaction — redact personal data of natural persons in scanned PDFs.

Pure LLM pipeline (vision-model visual grounding via OpenRouter). No OCR.
Company names and company contact data stay visible.

Pipeline: rasterize -> parallel detection (full page + deep crops) ->
self-verification -> burn -> rebuild -> verification layer
(pixel-diff vs. original + LLM leak audit + auto-fix).

Usage:
  python redact_pdf.py input.pdf
  python redact_pdf.py input.pdf -o out.pdf --dpi 200 --workers 6
  python redact_pdf.py input.pdf --no-verify          # skip verification layer
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redactor.config import DEFAULT_MODEL, get_api_key  # noqa: E402
from redactor.pipeline import redact_pdf  # noqa: E402
from redactor.verify import run_verification  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="scanned PDF to redact")
    ap.add_argument("-o", "--output", help="output PDF (default: <input>_redacted.pdf)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter vision model (default: {DEFAULT_MODEL})")
    ap.add_argument("--audit-model", default=None, help="model for the verification audit (default: same as --model)")
    ap.add_argument("--dpi", type=int, default=150, help="rasterization DPI (default 150)")
    ap.add_argument("--passes", type=int, default=1, help="extra self-verification passes (default 1)")
    ap.add_argument("--pad", type=int, default=6, help="box padding in px (default 6)")
    ap.add_argument("--workers", type=int, default=4, help="parallel pages / API calls (default 4)")
    ap.add_argument("--review-dir", help="write per-page QA overlay PNGs here (default: <output>_review)")
    ap.add_argument("--report-text", action="store_true", help="include redacted text spans in report JSON (sensitive!)")
    ap.add_argument("--no-deep", action="store_true", help="skip the extra half-page detection passes (faster, less thorough)")
    ap.add_argument("--no-verify", action="store_true", help="skip the verification layer (pixel diff + leak audit)")
    ap.add_argument("--no-fix", action="store_true", help="verification only reports leaks, does not auto-fix them")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"input not found: {src}")
    out = Path(args.output) if args.output else src.with_name(src.stem + "_redacted.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    review = args.review_dir or str(out.with_suffix("")) + "_review"

    key = get_api_key()
    print(f"Redacting {src.name} -> {out}")
    print(f"model={args.model}  dpi={args.dpi}  passes={args.passes}  workers={args.workers}")
    t0 = time.time()
    report = redact_pdf(str(src), str(out), args.model, key,
                        dpi=args.dpi, verify_passes=args.passes, pad=args.pad,
                        review_dir=review, report_text=args.report_text,
                        deep=not args.no_deep, workers=args.workers)
    print(f"── detection: {report['total_boxes']} boxes in {report['elapsed_sec']}s")

    if not args.no_verify:
        boxes_by_page = {p["page"]: [b["bbox_px"] for b in p["boxes"]] for p in report["pages"]}
        print("── verification layer (pixel diff + leak audit)...")
        v = run_verification(str(src), str(out), boxes_by_page, args.audit_model or args.model, key,
                             fix=not args.no_fix, workers=args.workers, review_dir=review)
        report["verification"] = v
        if v.get("auto_fixed"):
            report["total_boxes"] = sum(len(b) for b in v["final_boxes_by_page"].values())
            for pg in report["pages"]:
                pg["boxes"] = [{"bbox_px": [round(x, 1) for x in b], "category": "audit_fix"}
                               for b in v["final_boxes_by_page"].get(str(pg["page"]), [])]
        Path(out).with_suffix(".report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        diff_bad = [d for d in v["pixel_diff"] if d["burn_failures"] or d["unexpected_regions"]]
        print(f"   pixel diff: {'OK' if not diff_bad else 'ANOMALIES: ' + json.dumps(diff_bad[:3])}")
        if v.get("auto_fixed"):
            for pg, items in v["auto_fixed"].items():
                for it in items:
                    print(f"   🔧 auto-fixed p{pg}: [{it['kind']}] {it['text']!r}")
        if v["final_leaks"]:
            print(f"   ⚠️ REMAINING LEAKS (need human review): {json.dumps(v['final_leaks'], ensure_ascii=False)}")
        else:
            print("   leak audit: CLEAN")
        print(f"   verification: {'✅ OK' if v['ok'] else '⚠️ see report'} ({v['elapsed_sec']}s)")

    print(f"\n✅ {out}  ({report['total_boxes']} boxes, {time.time()-t0:.0f}s total)")
    print(f"   QA overlays: {review}/   report: {out.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
