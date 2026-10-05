#!/usr/bin/env python3
"""pdf-redact-web — browser UI to review & adjust LLM redactions before burning.

Flow:  upload PDF -> pages rasterized -> AI detection (reuses no-ocr-redaction
engine: full-page + deep half-page passes, Qwen3-VL) -> UI shows every box ->
user toggles boxes off/on, draws custom rectangles -> export burns ONLY the
kept boxes into an image-only PDF (irreversible, 0 extractable chars) ->
optional LLM leak audit of the burned result.

Run:  python3 web/server.py   (http://localhost:8799)
"""
import io
import os
import sys
import threading
import time
import uuid
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "engine"))  # shared redactor engine

import pymupdf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from redactor.config import DEFAULT_MODEL, get_api_key
from redactor import pipeline as P
from redactor import llm as L
from redactor.llm import usage_snapshot, usage_diff
from redactor.pdfops import render_page, norm_to_px, pad_box, burn, rebuild_pdf
from redactor import verify as V

DPI = 150
PORT = int(os.environ.get("REDACT_PORT", "8799"))
JOBS_DIR = BASE / "jobs"; JOBS_DIR.mkdir(exist_ok=True)
# redacted exports: REDACT_OUT_DIR env var, default <project>/redacted
OUT_DIR = Path(os.environ.get("REDACT_OUT_DIR") or (BASE.parent / "redacted"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS: dict = {}

app = FastAPI(title="pdf-redact-web")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def _job(jid: str) -> dict:
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[jid]


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


# ---------------- upload ----------------
@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    jid = uuid.uuid4().hex[:12]
    jd = JOBS_DIR / jid; jd.mkdir(parents=True)
    src = jd / "original.pdf"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        doc = pymupdf.open(src)
    except Exception as e:
        raise HTTPException(400, f"not a valid PDF: {e}")
    npages = len(doc)

    def raster(p):
        render_page(doc[p], DPI).save(jd / f"page-{p+1}.jpg", "JPEG", quality=85)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(raster, range(npages)))
    doc.close()

    JOBS[jid] = {"id": jid, "dir": jd, "src": src, "npages": npages, "name": file.filename,
                 "state": "uploaded", "pages_done": 0, "boxes": {p: [] for p in range(1, npages + 1)},
                 "custom_boxes": [], "error": None, "model": DEFAULT_MODEL}
    save_job(JOBS[jid])
    return {"job_id": jid, "npages": npages, "name": file.filename}


def save_job(job):
    (job["dir"] / "job.json").write_text(__import__("json").dumps(
        {k: job.get(k) for k in ("id", "name", "npages", "state", "pages_done", "boxes", "custom_boxes",
                                 "error", "model", "usage_detect", "usage_audit")}), "utf-8")


def _load_jobs():
    for jd in JOBS_DIR.iterdir():
        f = jd / "job.json"
        if not f.is_file():
            continue
        try:
            import json as _j
            d = _j.loads(f.read_text("utf-8"))
            d["boxes"] = {int(k): v for k, v in d.get("boxes", {}).items()}  # JSON str keys -> int
            d.setdefault("custom_boxes", [])
            d.update(dir=jd, src=jd / "original.pdf")
            JOBS[d["id"]] = d
        except Exception:
            pass


# ---------------- detection ----------------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


@app.post("/api/detect/{jid}")
def detect(jid: str):
    job = _job(jid)
    if job["state"] == "detecting":
        return {"ok": True, "note": "already running"}
    key = get_api_key()
    job.update(state="detecting", pages_done=0, error=None,
               boxes={p: [] for p in range(1, job["npages"] + 1)})

    def work():
        base = usage_snapshot()
        try:
            doc = pymupdf.open(job["src"])
            # pymupdf documents are NOT thread-safe: render page images
            # sequentially in THIS thread, submitting each page to the detection
            # pool as soon as it is rasterized (rendering pipelines with
            # detection instead of blocking it)
            page_imgs = []

            def one(p):
                img = page_imgs[p]
                raw = P._detect(img, DEFAULT_MODEL, key)
                raw += P._detect_deep(img, DEFAULT_MODEL, key)
                raw = [P._widen_numeric(b) for b in raw]
                raw.sort(key=lambda b: -((b["bbox_2d"][2]-b["bbox_2d"][0])*(b["bbox_2d"][3]-b["bbox_2d"][1])))
                kept = []
                for b in raw:  # drop near-duplicates (full-page vs deep pass)
                    if all(_iou(b["bbox_2d"], k["bbox_2d"]) < 0.55 for k in kept):
                        kept.append(b)
                kept = P.drop_official(kept)   # deterministic guard: officials stay visible
                for i, b in enumerate(kept):
                    b.update(id=f"p{p+1}-{i}", page=p + 1, enabled=True)
                # FAST PATH: publish boxes right after detection so the UI
                # draws them immediately (user sees progress at detection speed)
                job["boxes"][p + 1] = kept
                job["pages_done"] += 1
                save_job(job)
                # POLICY REFEREE (background): strikes company/official spans;
                # the UI picks up the update on its next status poll
                final = P.classify_spans(page_imgs[p], kept, DEFAULT_MODEL, key)
                if len(final) != len(kept):
                    job["boxes"][p + 1] = final
                    save_job(job)

            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = []
                for p in range(len(doc)):
                    page_imgs.append(render_page(doc[p], DPI))
                    futs.append(ex.submit(one, p))
                doc.close()  # all pages rasterized; no concurrent doc access
                for f in futs:
                    f.result()
            job["state"] = "detected"
            job["usage_detect"] = usage_diff(usage_snapshot(), base)
            save_job(job)
        except Exception as e:
            job["state"] = "error"; job["error"] = str(e)
            job["usage_detect"] = usage_diff(usage_snapshot(), base)
            save_job(job)

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "model": DEFAULT_MODEL}


@app.get("/api/status/{jid}")
def status(jid: str):
    job = _job(jid)
    flat = [{"id": b["id"], "page": b["page"], "bbox": b["bbox_2d"], "category": b["category"],
             "text": b.get("text", ""), "enabled": b.get("enabled", True)}
            for p in range(1, job["npages"] + 1) for b in job["boxes"][p]]
    flat += [{"id": c["id"], "page": c["page"], "bbox": c["bbox"], "category": "manual",
              "text": "", "enabled": True, "custom": True} for c in job.get("custom_boxes", [])]
    return {"state": job["state"], "npages": job["npages"], "pages_done": job["pages_done"],
            "error": job["error"], "model": job["model"], "boxes": flat,
            "usage": {"detect": job.get("usage_detect"), "audit": job.get("usage_audit"),
                      "live": usage_snapshot()}}


@app.get("/api/page/{jid}/{n}")
def page_img(jid: str, n: int):
    job = _job(jid)
    f = job["dir"] / f"page-{n}.jpg"
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


# ---------------- custom boxes (persisted user rectangles) ----------------
@app.post("/api/custom/{jid}")
def set_custom(jid: str, body: dict):
    job = _job(jid)
    clean = []
    for c in body.get("custom", []):
        try:
            bb = c["bbox"]
            if len(bb) != 4:
                continue
            p = int(c["page"])
            bb = [max(0, min(1000, round(float(v)))) for v in bb]
            if 1 <= p <= job["npages"] and bb[0] < bb[2] and bb[1] < bb[3]:
                clean.append({"id": str(c.get("id", ""))[:48] or f"c{len(clean)}", "page": p, "bbox": bb})
        except Exception:
            pass
    job["custom_boxes"] = clean
    save_job(job)
    return {"ok": True, "custom_boxes": len(clean)}


# ---------------- export ----------------
@app.post("/api/export/{jid}")
async def export(jid: str, body: dict):
    job = _job(jid)
    enabled = set(body.get("enabled", []))
    custom = body.get("custom", [])
    run_audit = bool(body.get("audit", True))

    doc = pymupdf.open(job["src"])
    pages, rects = [], []
    for p in range(len(doc)):
        img = render_page(doc[p], DPI)
        px = []
        for b in job["boxes"][p + 1]:
            if b["id"] in enabled:
                px.append(pad_box(norm_to_px(b["bbox_2d"], img.size), 8, 6, 10, img.size))
        for c in [c for c in custom if c.get("page") == p + 1]:
            px.append(pad_box(norm_to_px(c["bbox"], img.size), 4, 4, 4, img.size))
        pages.append(burn(img, px))
        rects.append(doc[p].rect)
    doc.close()

    stem = Path(job["name"]).stem
    out = OUT_DIR / f"{stem}_redacted.pdf"
    rebuilt = rebuild_pdf(pages, rects)
    rebuilt.save(out, garbage=3, deflate=True)
    rebuilt.close()

    result = {"ok": True, "path": str(out), "pages": len(pages), "burned_boxes": len(enabled),
              "custom_boxes": len(custom), "audit": None}

    if run_audit:  # LLM leak audit on the BURNED pages (no auto-fix: user fixes in UI)
        key = get_api_key()
        base = usage_snapshot()
        red = pymupdf.open(out)

        def audit(p):
            try:
                return p + 1, V._audit_page(str(out), p, job["model"], key)
            except Exception as e:
                return p + 1, [{"category": "audit_error", "text": str(e)}]
        with ThreadPoolExecutor(max_workers=4) as ex:
            leaks = dict(ex.map(audit, range(len(red))))
        red.close()
        result["audit"] = {str(p): v for p, v in leaks.items() if v}
        job["usage_audit"] = usage_diff(usage_snapshot(), base)
        result["usage"] = {"audit": job["usage_audit"]}
        save_job(job)

    # security check: output must be image-only
    chk = pymupdf.open(out)
    result["extractable_chars"] = sum(len(pg.get_text()) for pg in chk)
    chk.close()
    return result


@app.get("/api/download/{jid}")
def download(jid: str):
    job = _job(jid)
    stem = Path(job["name"]).stem
    f = OUT_DIR / f"{stem}_redacted.pdf"
    if not f.is_file():
        raise HTTPException(404, "nothing exported yet")
    return FileResponse(f, media_type="application/pdf", filename=f.name)


_load_jobs()

if __name__ == "__main__":
    import uvicorn
    print(f"pdf-redact-web on http://localhost:{PORT}  (docs at /docs)")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
