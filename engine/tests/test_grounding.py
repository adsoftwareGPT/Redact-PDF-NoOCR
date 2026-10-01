#!/usr/bin/env python3
"""Deterministic grounding QA for the sample letter.

The sample PDF is digitally generated, so we KNOW where every string sits
(page.search_for gives exact rectangles). The PRODUCTION pipeline never
extracts text — this test uses the text layer ONLY to measure the LLM's
detection quality against ground truth.

Run:  python tests/test_grounding.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf
from redactor.config import DEFAULT_MODEL, get_api_key
from redactor.pipeline import _detect, _verify
from redactor.pdfops import _overlap, merge_boxes, norm_to_px, pad_box, render_page
from redactor.pipeline import _detect_deep, _widen_numeric

DPI = 150

# (label, must-be-redacted search strings, min coverage fraction)
PII = [
    ("recipient name",   ["Maximilian Anton Biedermeier"], 0.95),
    ("street address",   ["Am Lindenhain 7a"],             0.95),
    ("city",             ["90579 Langenzenn"],              0.95),
    ("salutation name",  ["Herr"],                          0.90),  # 'Sehr geehrter Herr Biedermeier,' line
    ("date of birth",    ["23.07.1985"],                    0.95),
    ("personnel number", ["PN-4711203"],                    0.95),
    ("IBAN",             ["DE89 3704 0044 0532 0130 00"],   0.95),
    ("contact person",   ["Schulze-Vohenstrauß"],           0.95),
    ("mobile number",    ["0171 / 555 88 42"],              0.95),
    ("private email",    ["katharina.sv@t-online.de"],     0.95),
    ("signatory name",   ["Thomas Friedrich Krüger"],       0.95),
    ("personal ID",      ["X-9931"],                        0.95),
]

# Company data that MUST REMAIN VISIBLE (max tolerated coverage 10%)
KEEP = [
    ("company name",     ["Musterbau GmbH & Co. KG"]),
    ("company street",   ["Industriestraße 14, 90402 Nürnberg"]),
    ("company phone",    ["0911 / 123 45 67"]),
    ("company email",    ["info@musterbau-gmbh.de"]),
]


def covered_fraction(rect, boxes_px):
    area = rect.get_area()
    if area <= 0:
        return 1.0
    inter = sum(_overlap([rect.x0, rect.y0, rect.x1, rect.y1], b) for b in boxes_px)
    return min(1.0, inter / area)


def main():
    doc = pymupdf.open(ROOT / "samples" / "sample_letter.pdf")
    page = doc[0]
    img = render_page(page, DPI)
    size = img.size
    scale = DPI / 72.0
    key = get_api_key()

    print(f"Detecting (model={DEFAULT_MODEL}) ...")
    boxes = _detect(img, DEFAULT_MODEL, key)
    boxes += _detect_deep(img, DEFAULT_MODEL, key)
    boxes = [_widen_numeric(b) for b in boxes]
    boxes_px = [pad_box(norm_to_px(b["bbox_2d"], size), 6, 2, 8, size) for b in boxes]
    missed = _verify(img, boxes_px, DEFAULT_MODEL, key)
    boxes += [_widen_numeric(m) for m in missed]
    boxes_px = merge_boxes([pad_box(norm_to_px(b["bbox_2d"], size), 6, 2, 8, size) for b in boxes])

    print(f"{len(boxes)} spans -> {len(boxes_px)} merged boxes\n")
    fails = 0
    print("── PII must be covered ──────────────────────────────")
    for label, needles, need in PII:
        rects = [r for needle in needles for r in page.search_for(needle)]
        if not rects and label == "salutation name":
            rects = [r for r in page.search_for("Biedermeier,")]
        if not rects:
            print(f"  ?? {label:18s} ground truth not found in sample — update test")
            continue
        rect = rects[0]
        rect = pymupdf.Rect(rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale)
        cov = covered_fraction(rect, boxes_px)
        ok = cov >= need
        fails += 0 if ok else 1
        print(f"  {'✅' if ok else '❌'} {label:18s} coverage {cov:5.0%} (need {need:.0%})")

    print("── Company data must stay visible ───────────────────")
    for label, needles in KEEP:
        rects = [r for needle in needles for r in page.search_for(needle)]
        if not rects:
            print(f"  ?? {label:18s} ground truth not found in sample — update test")
            continue
        rect = rects[0]
        rect = pymupdf.Rect(rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale)
        cov = covered_fraction(rect, boxes_px)
        ok = cov <= 0.10
        fails += 0 if ok else 1
        print(f"  {'✅' if ok else '❌'} {label:18s} covered {cov:5.0%} (tolerate ≤10%)")

    print("─────────────────────────────────────────────────────")
    print("RESULT:", "PASS ✅" if fails == 0 else f"FAIL ❌ ({fails} checks)")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
