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
    boxes = [_widen_numeric(b) for b in boxes]
    boxes_px = _pad_all(boxes, size, pad)

    for _ in range(verify_passes):
        if not boxes_px:
            break
        missed = _verify(img, boxes_px, model, key)
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
