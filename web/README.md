# pdf-redact-web

Browser UI for reviewing AI redactions before burning them — inspired by redact-pdf.ai.
Reuses the proven `no-ocr-redaction` engine (Qwen3-VL via OpenRouter, no OCR).

## Start
```bash
cd ~/../mnt/c/Users/Uwe.Heinig/Desktop/pdf-redact-web   # or the Windows path
python3 server.py            # → http://localhost:8799
```
Open http://localhost:8799 in your Windows browser (WSL localhost forwarding).

## Workflow
1. **Upload** a PDF (drop zone) → pages rasterized at 150 dpi
2. **AI detection** runs in background (full-page + deep half-page passes, 4 workers);
   boxes stream in per page; progress bar + category chips with counts
3. **Review**
   - filled dark box = will be redacted → **click to keep visible** (green dashed)
   - click again to re-enable
   - **drag anywhere** = draw your own rectangle (blue) — any number of them,
     overlapping allowed; remove via click on its **×** or double-click
   - category chip = toggle whole category on/off
4. **Export redacted PDF** → burns ONLY kept boxes + custom rectangles into an
   image-only PDF (0 extractable chars, irreversible), saved to `Desktop\redacted\`,
   then runs an LLM leak audit on the burned pages; findings shown in sidebar
   (fix by drawing a box and re-exporting — no re-detection cost)

## Notes
- Jobs persist in `jobs/<id>/` (survive server restarts; deep-link via `#<jobid>`)
- Custom rectangles are persisted server-side too (`POST /api/custom/<id>`) —
  they survive page reloads and server restarts
- Detection ≈ 10 s/page · export+audit ≈ 5 s/page (Qwen3-VL, few cents/page)
- API key auto-loaded from `C:\agent_linux\.env` (`OPENROUTER_KEY`)
- REST API under `/docs` (FastAPI swagger)
