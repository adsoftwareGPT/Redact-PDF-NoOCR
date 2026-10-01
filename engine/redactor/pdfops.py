"""PDF rasterization, box math, burning and rebuilding. No OCR, no text extraction."""
import io

import pymupdf
from PIL import Image, ImageDraw

NORM = 1000.0  # Qwen3-VL grounding coordinate space


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """Rasterize a PDF page (this is the ONLY thing we do with the PDF content)."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def norm_to_px(bbox_norm, size):
    """[x1,y1,x2,y2] 0-1000 -> pixel coords, with safety padding."""
    w, h = size
    x1, y1, x2, y2 = bbox_norm
    return [x1 / NORM * w, y1 / NORM * h, x2 / NORM * w, y2 / NORM * h]


def pad_box(box, pad_x: int, pad_top: int, pad_bottom: int, size):
    """Asymmetric padding: more room at the bottom (descenders + the model's
    systematic tendency to ground text boxes slightly too high)."""
    w, h = size
    x1, y1, x2, y2 = box
    return [max(0, x1 - pad_x), max(0, y1 - pad_top), min(w, x2 + pad_x), min(h, y2 + pad_bottom)]


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def merge_boxes(boxes_px, gap: int = 8):
    """Union boxes that overlap or lie within `gap` px of each other."""
    boxes = [list(b) for b in boxes_px]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop(0)
            grown = [a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap]
            i = 0
            while i < len(boxes):
                if _overlap(grown, boxes[i]) > 0:
                    b = boxes.pop(i)
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    grown = [a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap]
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


def dedupe(new_boxes, existing_px, size, iou_thresh=0.4):
    """Drop new normalized boxes that mostly overlap an already-known pixel box."""
    kept = []
    for nb in new_boxes:
        pxb = norm_to_px(nb["bbox_2d"], size)
        area = max(1e-6, (pxb[2] - pxb[0]) * (pxb[3] - pxb[1]))
        if all(_overlap(pxb, ex) / area < iou_thresh for ex in existing_px):
            kept.append(nb)
    return kept


def draw_review(img: Image.Image, boxes_px) -> Image.Image:
    """Translucent red overlay for human QA."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay)
    for b in boxes_px:
        dr.rectangle(b, fill=(220, 30, 30, 110), outline=(200, 0, 0, 255), width=2)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def burn(img: Image.Image, boxes_px) -> Image.Image:
    """Irreversibly paint solid black boxes."""
    out = img.copy()
    dr = ImageDraw.Draw(out)
    for b in boxes_px:
        dr.rectangle(b, fill=(0, 0, 0))
    return out


def rebuild_pdf(pages: list, page_rects: list, jpeg_quality=92) -> pymupdf.Document:
    """Build a fresh image-only PDF — original content objects are NOT carried over,
    so covered data cannot be recovered from the output."""
    doc = pymupdf.Document()
    for img, rect in zip(pages, page_rects):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        page = doc.new_page(width=rect.width, height=rect.height)
        page.insert_image(pymupdf.Rect(0, 0, rect.width, rect.height), stream=buf.getvalue())
    doc.set_metadata({})  # strip author/producer metadata
    return doc
