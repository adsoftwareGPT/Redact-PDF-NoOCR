"""Verification layer: compare the redacted copy against the ORIGINAL.

1. Pixel diff (local, deterministic, free)
   - confirms every intended box is actually burned black
   - flags unexpected large content changes (corruption / accidental loss)
2. LLM leak audit (per page, parallel)
   - fresh transcription-style read of each REDACTED page
   - anything personal still readable is reported with a bounding box
3. Auto-fix (optional)
   - actionable leaks are unioned into the burn set, the page is re-burned
     from the original and re-audited once
"""
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

from . import llm
from .pdfops import burn, draw_review, merge_boxes, norm_to_px, pad_box, render_page, rebuild_pdf

# Handwritten form words that are NOT personal identifiers — never auto-redact
FORM_WORDS = {"eigentum", "eigentümer", "verwalter", "vertreter", "vollmacht",
              "anwesend", "abwesend", "ja", "nein", "enthalten", "stimme",
              "eigennutzung", "vermietet"}

AUDIT_PROMPT = """Transcription audit of one REDACTED scanned page of a document (personal data was covered with solid black boxes). Read the ENTIRE page including tables, lists and margins.

Report everything below that is still READABLE and NOT covered by a black box:
- names or surnames of PRIVATE persons (even single surnames in tables, letterheads, footers)
- handwritten signatures or person initials (name-like pen marks) of private persons
- personal email addresses, private phone numbers, IBANs of private individuals

Do NOT report (EXEMPT by policy): ALL company data (names, addresses, contact data, register numbers); OFFICIAL PERSONS acting in their official/business capacity — notaries (Notar/Notarin) and notary staff, judges and court employees (Richterin, Rechtspfleger ...), employees of companies/courts/authorities named with a function label or in the letterhead/signature block (incl. their titles and handwritten signatures); printed column headers / form labels (Eigentum, Eigentümer, Verwalter, HGV ...), handwritten form words (Eigentum/Eigentümer/Verwalter as representation notes), X marks in checkboxes, or the black boxes.

Return STRICT JSON only:
{"leaks": [{"bbox_2d": [x1,y1,x2,y2], "kind": "name|signature|email|phone|bank", "text": "<readable span>"}]}
Coordinates: integers 0-1000 normalized to image width/height. Nothing leaks: {"leaks": []}."""

AUDIT_DPI = 100   # pixel diff resolution
BLOCK = 12        # diff block size in px (coarse = robust to JPEG noise)
DIFF_T = 65       # gray-level difference threshold per pixel
HOT_T = 30        # avg block value that marks a block as changed


# ── 1. pixel diff ───────────────────────────────────────────────────────────
def _render(pdf, pno, dpi):
    return Image.open(io.BytesIO(pdf[pno].get_pixmap(dpi=dpi).tobytes("png"))).convert("L")


def _diff_regions(a: Image.Image, b: Image.Image):
    """Coarse changed-block clusters between two same-size grayscale renders."""
    if a.size != b.size:
        b = b.resize(a.size)
    diff = ImageChops.difference(a, b).point(lambda v: 255 if v > DIFF_T else 0)
    bw, bh = a.size[0] // BLOCK, a.size[1] // BLOCK
    blocks = diff.resize((max(1, bw), max(1, bh)), Image.BOX)
    data = blocks.getdata()
    w = blocks.size[0]
    hot = {(i % w, i // w) for i, v in enumerate(data) if v > HOT_T}
    # greedy clustering of adjacent hot blocks
    clusters = []
    while hot:
        seed = hot.pop()
        stack, cluster = [seed], {seed}
        while stack:
            x, y = stack.pop()
            for nb in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if nb in hot:
                    hot.remove(nb); cluster.add(nb); stack.append(nb)
        clusters.append(cluster)
    regions = []
    for c in clusters:
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        regions.append([min(xs)*BLOCK, min(ys)*BLOCK, (max(xs)+1)*BLOCK, (max(ys)+1)*BLOCK])
    return regions, len(hot) == 0


def pixel_diff(original_path, redacted_path, boxes_by_page, dpi=AUDIT_DPI):
    """Per-page: changed regions + burn check + unexpected-change check."""
    orig, red = pymupdf.open(original_path), pymupdf.open(redacted_path)
    out = []
    scale = dpi / 150.0  # boxes_by_page coords are px @150dpi
    for pno in range(len(red)):
        a, b = _render(orig, pno, dpi), _render(red, pno, dpi)
        regions, _ = _diff_regions(a, b)
        area = a.size[0] * a.size[1]
        changed_px = sum((r[2]-r[0]) * (r[3]-r[1]) for r in regions)

        # burn check: each intended box's centre must be black in the redacted render
        burn_failures = []
        for i, box in enumerate(boxes_by_page.get(pno + 1, [])):
            bx = [v * scale for v in box]
            w_in, h_in = max(1, int((bx[2]-bx[0])*0.2)), max(1, int((bx[3]-bx[1])*0.2))
            samples = [((bx[0]+bx[2])//2, (bx[1]+bx[3])//2),
                       (int(bx[0]+w_in), int(bx[1]+h_in)), (int(bx[2]-w_in), int(bx[1]+h_in)),
                       (int(bx[0]+w_in), int(bx[3]-h_in)), (int(bx[2]-w_in), int(bx[3]-h_in))]
            black_hits = 0
            for cx, cy in samples:
                cx, cy = max(0, min(b.size[0]-1, cx)), max(0, min(b.size[1]-1, cy))
                if b.getpixel((cx, cy)) <= 60:
                    black_hits += 1
            if black_hits < 3:
                burn_failures.append({"box": i, "bbox_px": box})

        # unexpected changes: clusters far from any intended box
        pad = 30
        unexpected = []
        min_area = 0.015 * area          # >= 1.5% of the page: real anomalies only
        for r in regions:
            rw = (r[2]-r[0]) * (r[3]-r[1])
            if rw < min_area:            # ignore specks / JPEG-noise clusters
                continue
            rs = [v / scale for v in r]
            inside = any(not (rs[2] < b2[0]-pad or rs[0] > b2[2]+pad or
                              rs[3] < b2[1]-pad or rs[1] > b2[3]+pad)
                         for b2 in boxes_by_page.get(pno + 1, []))
            if not inside:
                unexpected.append(r)
        out.append({"page": pno + 1, "changed_area_pct": round(100*changed_px/area, 1),
                    "regions": len(regions), "burn_failures": burn_failures,
                    "unexpected_regions": unexpected})
    return out


def derive_boxes_from_diff(original_path, redacted_path, dpi=150, min_area=800):
    """Recover the effective redaction boxes by diffing original vs. redacted.
    Useful when a report is missing (e.g. crashed run) — the diff IS the truth."""
    orig, red = pymupdf.open(original_path), pymupdf.open(redacted_path)
    out = {}
    for pno in range(len(red)):
        a = _render(orig, pno, dpi)
        b = Image.open(io.BytesIO(red[pno].get_pixmap(dpi=dpi).tobytes("png"))).convert("L")
        regions, _ = _diff_regions(a, b)
        kept = [r for r in regions if (r[2]-r[0]) * (r[3]-r[1]) >= min_area]
        if kept:
            out[pno + 1] = kept
    return out


# ── 2+3. LLM audit + auto-fix ───────────────────────────────────────────────
def _page_b64(pdf, pno, dpi=150, fmt="JPEG"):
    buf = io.BytesIO()
    Image.open(io.BytesIO(pdf[pno].get_pixmap(dpi=dpi).tobytes("png"))).save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _audit_page(pdf_path, pno, model, key):
    pdf = pymupdf.open(pdf_path)
    raw = llm.call_vision_model(_page_b64(pdf, pno), AUDIT_PROMPT, model, key)
    return llm.validate_boxes([{**l, "category": l.get("kind", "other")}
                               for l in llm.extract_json(raw).get("leaks", [])])


def _actionable(leaks):
    keep, seen = [], set()
    for l in leaks:
        txt = str(l.get("text", "")).strip().lower().rstrip(":.,;")
        # normalize mojibake (models sometimes emit utf-8-read-as-latin1)
        txt = (txt.replace("Ã¼", "ü").replace("Ã¶", "ö").replace("Ã¤", "ä")
                  .replace("ÃŸ", "ß").replace("Ã„", "Ä").replace("Ã–", "Ö").replace("Ãœ", "Ü"))
        if txt in FORM_WORDS or txt.startswith("verwalt") or txt.startswith("vertret"):
            continue
        if txt in seen:               # same span reported twice on a page
            continue
        seen.add(txt)
        keep.append(l)
    return keep


def run_verification(original_path, redacted_path, boxes_by_page, model, api_key,
                     fix=True, workers=4, review_dir=None, quiet=False, dpi=150):
    t0 = time.time()
    verification = {"pixel_diff": pixel_diff(original_path, redacted_path, boxes_by_page),
                    "audit_model": model, "auto_fix": fix, "rounds": [], "final_leaks": {}}

    orig = pymupdf.open(original_path)
    red = pymupdf.open(redacted_path)
    npages = len(red)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_audit_page, redacted_path, p, model, api_key): p for p in range(npages)}
        leaks_by_page = {p: f.result() for f, p in futs.items()}
    verification["rounds"].append({"round": 1, "leaks": {p+1: [{"kind": l["category"], "text": l["text"]} for l in v]
                                                          for p, v in leaks_by_page.items() if v}})

    fixes, remaining = {}, {}
    for p, leaks in leaks_by_page.items():
        act = _actionable(leaks)
        if act:
            fixes[p] = act
        elif leaks:
            remaining[p + 1] = [{"kind": l["category"], "text": l["text"]} for l in leaks]

    if fixes and fix:
        # re-burn fixed pages from the ORIGINAL render + union boxes
        pages, rects = [], []
        boxes_final = {}
        for pno in range(npages):
            o_img = render_page(orig[pno], dpi)
            boxes = list(boxes_by_page.get(pno + 1, []))
            if pno in fixes:
                extra = [pad_box(norm_to_px(l["bbox_2d"], o_img.size), 8, 4, 10, o_img.size)
                         for l in fixes[pno]]
                boxes = merge_boxes(boxes + extra)
                if review_dir:
                    rdir = Path(review_dir); rdir.mkdir(parents=True, exist_ok=True)
                    draw_review(o_img, boxes).save(rdir / f"page_{pno+1:03d}.png")
            boxes_final[pno + 1] = boxes
            pages.append(burn(o_img, boxes))
            rects.append(orig[pno].rect)
        rebuild_pdf(pages, rects).save(redacted_path + ".tmp", garbage=4, deflate=True)
        Path(redacted_path + ".tmp").replace(redacted_path)

        # one re-audit of the fixed pages only
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_audit_page, redacted_path, p, model, api_key): p for p in fixes}
            recheck = {p: f.result() for f, p in futs.items()}
        for p, leaks in recheck.items():
            if leaks:
                remaining[p + 1] = [{"kind": l["category"], "text": l["text"]} for l in _actionable(leaks)]
        verification["rounds"].append({"round": 2, "rechecked_pages": [p+1 for p in fixes],
                                       "leaks": {p+1: [{"kind": l["category"], "text": l["text"]} for l in v]
                                                 for p, v in recheck.items() if v}})
        verification["auto_fixed"] = {p + 1: [{"kind": l["category"], "text": l["text"]} for l in v]
                                      for p, v in fixes.items()}
        verification["final_boxes_by_page"] = {str(k): v for k, v in boxes_final.items()}

    verification["final_leaks"] = remaining
    verification["elapsed_sec"] = round(time.time() - t0, 1)
    verification["ok"] = not remaining and not any(
        r["burn_failures"] or r["unexpected_regions"] for r in verification["pixel_diff"])
    return verification
