"""End-to-end pipeline: rasterize -> parallel LLM detection -> verification
pass -> burn -> rebuild. Page processing is parallelized; detection calls
(full page + deep crops) run concurrently per page."""
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from . import llm, prompts
from .pdfops import (burn, draw_review, merge_boxes, norm_to_px, pad_box,
                     render_page, rebuild_pdf)

# Categories where models systematically under-box long digit sequences
_NUMERIC_CATS = ("bank", "iban", "account", "id_number", "personnel", "social",
                 "tax", "phone", "fax", "dob", "date_of_birth", "insurance")


def _widen_numeric(b, frac=0.10, cap=40):
    """Widen numeric-category boxes (IBAN etc.) — vision models tend to cut off
    trailing digit groups. Over-redacts a few px of adjacent text; safe side."""
    cat = str(b.get("category", "")).lower()
    if not any(k in cat for k in _NUMERIC_CATS):
        return b
    x1, y1, x2, y2 = b["bbox_2d"]
    ext = min(cap, (x2 - x1) * frac)
    return {**b, "bbox_2d": [max(0.0, x1 - ext), y1, min(1000.0, x2 + ext), y2]}


def _img_b64(img, fmt="JPEG", quality=85):
    """JPEG transport: ~10x smaller uploads than PNG -> faster round-trips."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return base64.b64encode(buf.getvalue()).decode(), f"image/{fmt.lower()}"


def _detect(img, model, key):
    b64, mime = _img_b64(img)
    raw = llm.call_vision_model(b64, prompts.DETECT_PROMPT, model, key, mime=mime)
    return llm.validate_boxes(llm.extract_json(raw).get("boxes", []))


def _detect_band(img, y0, y1, model, key):
    """Detection on one horizontal band [y0,y1) px; coords mapped back to page."""
    w, h = img.size
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if y1 - y0 < 40:
        return []
    crop = img.crop((0, y0, w, y1))
    b64, mime = _img_b64(crop)
    raw = llm.call_vision_model(b64, prompts.DETECT_PROMPT, model, key, mime=mime)
    out = []
    for b in llm.validate_boxes(llm.extract_json(raw).get("boxes", [])):
        x1, by1, x2, by2 = b["bbox_2d"]
        out.append({**b, "bbox_2d": [
            x1, (y0 + by1 / 1000.0 * (y1 - y0)) / h * 1000.0,
            x2, (y0 + by2 / 1000.0 * (y1 - y0)) / h * 1000.0]})
    return out


def _detect_deep(img, model, key, bands=2, overlap=0.25, executor=None):
    """Deep pass on overlapping horizontal crops (legacy API: sequential)."""
    w, h = img.size
    band_h = int(h / (bands - overlap * (bands - 1)))
    step = int(band_h * (1 - overlap))
    spans = []
    for i in range(bands):
        y0 = min(i * step, max(0, h - band_h))
        y1 = min(y0 + band_h, h)
        spans.append((y0, y1))
    return _detect_bands(img, spans, model, key, executor)


def _detect_bands(img, spans, model, key, executor=None):
    if executor is None:
        return [b for y0, y1 in spans for b in _detect_band(img, y0, y1, model, key)]
    futs = [executor.submit(_detect_band, img, y0, y1, model, key) for y0, y1 in spans]
    return [b for f in futs for b in f.result()]


def _verify(img, boxes_px, model, key):
    """Second opinion on the FINAL output: solid black boxes burned in, so the
    model sees exactly what a recipient would see — partial overhangs (e.g. the
    last digits of an IBAN sticking out) become obvious."""
    marked = burn(img, boxes_px)
    b64, mime = _img_b64(marked)
    raw = llm.call_vision_model(b64, prompts.VERIFY_PROMPT, model, key, mime=mime)
    return llm.validate_boxes(llm.extract_json(raw).get("missed", []))


def _pad_all(boxes, size, pad):
    px_pad = (pad, max(2, pad // 3), max(5, pad + 2))
    return [pad_box(norm_to_px(b["bbox_2d"], size), *px_pad, size) for b in boxes]


def _process_page(page, pno, npages, model, key, dpi, verify_passes, pad, deep, quiet):
    img = render_page(page, dpi)
    w, h = img.size
    size = img.size

    # detection: full page + deep bands run CONCURRENTLY
    spans = None
    if deep:
        h = size[1]
        band_h = int(h / (2 - 0.25))
        step = int(band_h * 0.75)
        spans = [(min(0 * step, h - band_h), min(band_h, h)),
                 (min(1 * step, max(0, h - band_h)), min(step + band_h, h))]
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_full = ex.submit(_detect, img, model, key)
        f_bands = ex.submit(_detect_bands, img, spans, model, key, ex) if spans else None
        boxes = f_full.result() + (f_bands.result() if f_bands else [])
    boxes = drop_official([_widen_numeric(b) for b in boxes])
    boxes = classify_spans(img, boxes, model, key, quiet=quiet)
    boxes_px = _pad_all(boxes, size, pad)

    for _ in range(verify_passes):
        if not boxes_px:
            break
        missed = drop_official(_verify(img, boxes_px, model, key))
        if not missed:
            break
        boxes.extend(_widen_numeric(m) for m in missed)
        boxes_px = _pad_all(boxes, size, pad)

    merged = merge_boxes(boxes_px)
    if not quiet:
        print(f"  page {pno}/{npages}: {len(merged)} redaction box(es)", flush=True)
    return {"page": pno, "img": img, "rect": page.rect, "merged": merged,
            "boxes": boxes, "size": size, "failed": False}


def _process_page_safe(*a, **kw):
    """Fail-safe wrapper: one retry, then full-page blackout (never leak)."""
    try:
        return _process_page(*a, **kw)
    except Exception as e:
        print(f"  ⚠️ page {a[1]} failed ({type(e).__name__}: {str(e)[:80]}), retrying...", flush=True)
        try:
            return _process_page(*a, **kw)
        except Exception as e2:
            print(f"  ⛔ page {a[1]} failed twice — FULL-PAGE BLACKOUT ({type(e2).__name__})", flush=True)
            img = render_page(a[0], a[4] if False else kw.get("dpi", a[4]))
            from PIL import Image, ImageDraw
            black = Image.new("RGB", img.size, (0, 0, 0))
            return {"page": a[1], "img": black, "rect": a[0].rect,
                    "merged": [[0, 0, img.size[0], img.size[1]]], "boxes": [],
                    "size": img.size, "failed": True}


def redact_pdf(input_path: str, output_path: str, model: str, api_key: str,
               dpi: int = 150, verify_passes: int = 1, pad: int = 6,
               review_dir: str = None, report_text: bool = False,
               quiet: bool = False, deep: bool = True, workers: int = 4):
    """Returns the report dict. Writes output PDF (+ optional review PNGs)."""
    t_start = time.time()
    src = pymupdf.open(input_path)
    if src.needs_pass:
        raise SystemExit(f"{input_path}: password-protected PDFs are not supported")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_process_page_safe, page, i + 1, len(src), model, api_key,
                          dpi, verify_passes, pad, deep, quiet): i
                for i, page in enumerate(src)}
        results = [None] * len(src)
        for f, i in futs.items():
            results[i] = f.result()
    src.close()

    out_pages = [burn(r["img"], r["merged"]) for r in results]
    out_doc = rebuild_pdf(out_pages, [r["rect"] for r in results])
    # save-to-temp, then robust handover: the target may be LOCKED by a
    # Windows viewer (drvfs) — retry, then copy, then fall back to a new name
    out_doc.save(output_path + ".tmp", garbage=4, deflate=True)
    _robust_finalize(output_path + ".tmp", output_path)

    if review_dir:
        rdir = Path(review_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        for r in results:
            draw_review(r["img"], r["merged"]).save(rdir / f"page_{r['page']:03d}.png")

    report_pages = []
    for r in results:
        entry = {"page": r["page"], "size": list(r["size"]),
                 "boxes": [{"bbox_px": [round(v, 1) for v in m],
                            "category": _first_cat(m, r["boxes"], r["size"])} for m in r["merged"]]}
        if report_text:
            entry["spans"] = [{"bbox_norm": b["bbox_2d"], "category": b["category"], "text": b["text"]} for b in r["boxes"]]
        report_pages.append(entry)

    report = {
        "file": str(input_path), "output": str(output_path), "model": model,
        "dpi": dpi, "verify_passes": verify_passes, "pad_px": pad,
        "deep": deep, "workers": workers,
        "pages": report_pages, "total_boxes": sum(len(r["merged"]) for r in results),
        "failed_pages": [r["page"] for r in results if r.get("failed")],
        "elapsed_sec": round(time.time() - t_start, 1),
        "note": "output rebuilt from rasterized pages; covered content is destroyed, not hidden",
    }
    Path(output_path).with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _robust_finalize(tmp_path, target_path, retries=3, wait=2.0):
    """Replace target with tmp, tolerating Windows-side file locks."""
    import os
    import shutil
    import time as _t
    for i in range(retries):
        try:
            os.replace(tmp_path, target_path)
            return target_path
        except PermissionError:
            if i < retries - 1:
                _t.sleep(wait)
    try:  # copy into place (works if target is writable but not replaceable)
        shutil.copyfile(tmp_path, target_path)
        os.unlink(tmp_path)
        return target_path
    except OSError:
        alt = str(target_path)[:-4] + "_new.pdf"
        shutil.copyfile(tmp_path, alt)
        os.unlink(tmp_path)
        print(f"  ⚠️ {target_path} is LOCKED (close the PDF viewer!) — saved as {alt}", flush=True)
        return alt


def _first_cat(merged_box, boxes, size):
    """Category of the span with the LARGEST overlap inside a merged box."""
    from .pdfops import _overlap
    best, best_ov = "other", 0.0
    for b in boxes:
        pxb = norm_to_px(b["bbox_2d"], size)
        ov = _overlap(pxb, merged_box)
        if ov > best_ov:
            best, best_ov = b["category"], ov
    return best


# ── deterministic guard: official persons are never redacted ──────────────
# High-precision German role words — a detection box whose text contains one
# is a notary / court / company functionary and is ALWAYS kept visible
# (policy: only PRIVATE natural persons get redacted).
import re as _re
OFFICIAL_ROLE_RE = _re.compile(
    r"(?i)\b(notariat|notarassessor(in)?|notar(in)?v?e?r?t?r?e?t?e?r?(in)?|notar(in)?|"
    r"rechtspfleger(in)?|urkundsbeamte[rn]|richter(in)?|amtsrichter(in)?|"
    r"prokurist(in)?|objektmanager(in)?|kundenberater(in)?|sachbearbeiter(in)?|"
    r"geschaeftsfuehrer(in)?|geschäftsführer(in)?)\b"
    r"|\bi\.\s?[AV]\b")


STAFF_DIAL_RE = _re.compile(r"(?i)durchwahl|direktwahl|extension|dw[-:]?\s*\d|[-\u2013]\s?\d{3}\s?\)")


def _inside(b, big, frac=0.9):
    """Is box b at least `frac` contained in box `big`? (normalized coords)"""
    x1, y1, x2, y2 = b["bbox_2d"]; X1, Y1, X2, Y2 = big["bbox_2d"]
    iw = max(0, min(x2, X2) - max(x1, X1)); ih = max(0, min(y2, Y2) - max(y1, Y1))
    area = (x2 - x1) * (y2 - y1)
    return area > 0 and (iw * ih) / area >= frac


def drop_official(boxes):
    """Remove boxes that cover official-person spans (notaries, court and
    company functionaries, staff named with a business direct dial). Tight
    boxes fully inside a dropped span inherit the verdict. Fail-safe for the
    keep-direction only: a drop never removes evidence of a PRIVATE person,
    because drops require an explicit business/official cue."""
    fixed = []
    for b in boxes:
        t = str(b.get("text", "")).replace("Ã¼", "ü").replace("Ã¶", "ö").replace("Ã¤", "ä")
        if OFFICIAL_ROLE_RE.search(t) or STAFF_DIAL_RE.search(t):
            continue
        fixed.append(b)
    # inheritance: sub-boxes inside dropped boxes share the official context
    dropped = [b for b in boxes if b not in fixed]
    return [b for b in fixed if not any(_inside(b, d) for d in dropped)]


# ── policy referee: per-span context-crop keep/redact verdict ──────────────
def _span_crop(img, b, ctx=0.45, min_w=420, min_h=140):
    """Crop around a normalized box with generous context margins."""
    w, h = img.size
    x1, y1, x2, y2 = b["bbox_2d"]
    px1, py1 = x1 / 1000 * w, y1 / 1000 * h
    px2, py2 = x2 / 1000 * w, y2 / 1000 * h
    bw, bh = px2 - px1, py2 - py1
    cw, ch = max(bw * (1 + 2 * ctx), min_w), max(bh * (1 + 2.5 * ctx), min_h)
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    X1, Y1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    X2, Y2 = min(w, int(cx + cw / 2)), min(h, int(cy + ch / 2))
    return img.crop((X1, Y1, X2, Y2)), (px1 - X1, py1 - Y1, px2 - X1, py2 - Y1)


def _verdict_one(img, b, model, key):
    """One span -> ('redact'|'keep'). Fail-closed: errors mean redact."""
    from PIL import ImageDraw
    crop, rel = _span_crop(img, b)
    dr = ImageDraw.Draw(crop)
    dr.rectangle(rel, outline=(255, 0, 0), width=4)
    b64, mime = _img_b64(crop)
    raw = llm.call_vision_model(b64, prompts.SPAN_VERDICT_PROMPT, model, key, mime=mime)
    v = llm.extract_json(raw).get("verdict", "redact")
    return str(v).lower() if str(v).lower() in ("keep", "redact") else "redact"


def _verdict_one_safe(img, b, model, key):
    """Per-box fail-closed wrapper: one API hiccup must not fail the whole pass."""
    try:
        return _verdict_one(img, b, model, key)
    except Exception:
        return "redact"


def classify_spans(img, boxes, model, key, quiet=True, workers=6):
    """Context-aware referee: strikes company/official spans over-marked by the
    detector. Each candidate is judged on its own crop (surrounding text gives
    the role/company context). Fail-closed: unparseable -> redact."""
    if not boxes:
        return boxes
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(lambda b: _verdict_one_safe(img, b, model, key), boxes))
    keep = [b for b, v in zip(boxes, verdicts) if v == "redact"]
    if not quiet:
        print(f"    referee: {len(boxes) - len(keep)}/{len(boxes)} span(s) stay visible", flush=True)
    return keep


# ── deterministic guard: official persons are never redacted ──────────────
# High-precision German role words — a detection box whose text contains one
# is a notary / court / company functionary and is ALWAYS kept visible
# (policy: only PRIVATE natural persons get redacted).
import re as _re
OFFICIAL_ROLE_RE = _re.compile(
    r"(?i)\b(notariat|notarassessor(in)?|notar(in)?v?e?r?t?r?e?t?e?r?(in)?|notar(in)?|"
    r"rechtspfleger(in)?|urkundsbeamte[rn]|richter(in)?|amtsrichter(in)?|"
    r"prokurist(in)?|objektmanager(in)?|kundenberater(in)?|sachbearbeiter(in)?|"
    r"geschaeftsfuehrer(in)?|geschäftsführer(in)?)\b"
    r"|\bi\.\s?[AV]\b")


STAFF_DIAL_RE = _re.compile(r"(?i)durchwahl|direktwahl|extension|dw[-:]?\s*\d|[-\u2013]\s?\d{3}\s?\)")


def _inside(b, big, frac=0.9):
    """Is box b at least `frac` contained in box `big`? (normalized coords)"""
    x1, y1, x2, y2 = b["bbox_2d"]; X1, Y1, X2, Y2 = big["bbox_2d"]
    iw = max(0, min(x2, X2) - max(x1, X1)); ih = max(0, min(y2, Y2) - max(y1, Y1))
    area = (x2 - x1) * (y2 - y1)
    return area > 0 and (iw * ih) / area >= frac


def drop_official(boxes):
    """Remove boxes that cover official-person spans (notaries, court and
    company functionaries, staff named with a business direct dial). Tight
    boxes fully inside a dropped span inherit the verdict. Fail-safe for the
    keep-direction only: a drop never removes evidence of a PRIVATE person,
    because drops require an explicit business/official cue."""
    fixed = []
    for b in boxes:
        t = str(b.get("text", "")).replace("Ã¼", "ü").replace("Ã¶", "ö").replace("Ã¤", "ä")
        if OFFICIAL_ROLE_RE.search(t) or STAFF_DIAL_RE.search(t):
            continue
        fixed.append(b)
    # inheritance: sub-boxes inside dropped boxes share the official context
    dropped = [b for b in boxes if b not in fixed]
    return [b for b in fixed if not any(_inside(b, d) for d in dropped)]


# ── policy referee: context-aware keep/redact verdict per candidate span ───
def classify_spans(img, boxes, model, key, quiet=True):
    """One extra LLM call with the FULL page as context: strikes company /
    official-person spans the detector over-marked. Fail-closed: on any error
    or unparseable answer all boxes stay on the redaction list."""
    if not boxes:
        return boxes

    def _fix(t):  # repair mojibake so the model reads clean German
        return (str(t).replace("Ã¼", "ü").replace("Ã¶", "ö").replace("Ã¤", "ä")
                .replace("ÃŸ", "ß").replace("Ã", "Ü").replace("Ã", "Ö")
                .replace("Ã", "Ä"))

    listing = "\n".join(f"{i+1} · {b.get('category','?')} · {_fix(b.get('text','')).strip()[:120]}"
                        for i, b in enumerate(boxes))
    prompt = prompts.CLASSIFY_PROMPT.replace("{spans}", listing)
    try:
        b64, mime = _img_b64(img)
        raw = llm.call_vision_model(b64, prompt, model, key, mime=mime)
        verdicts = llm.extract_json(raw)
        if not isinstance(verdicts, dict):
            raise ValueError("verdict map is not an object")
        keep_idx = set()
        for k, v in verdicts.items():
            if str(k).strip().isdigit() and str(v).strip().lower() == "keep":
                keep_idx.add(int(k))
        if not quiet:
            print(f"    referee: {len(keep_idx)}/{len(boxes)} span(s) stay visible", flush=True)
        return [b for i, b in enumerate(boxes) if i + 1 not in keep_idx]
    except Exception as e:
        if not quiet:
            print(f"  classify pass failed ({type(e).__name__}) -> keeping all boxes", flush=True)
        return boxes
