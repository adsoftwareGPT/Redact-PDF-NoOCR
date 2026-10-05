# no-ocr-redaction

Redacts personal data of **PRIVATE natural persons** in scanned PDF documents —
**purely LLM-based, no OCR anywhere in the pipeline**. Company data stays visible,
as do **official persons** in their function (notaries, court staff, employees of
companies/courts/authorities acting in a business role). See the root README for
the full policy.

## How it works

```
scanned PDF ──rasterize──▶ page image ──▶ ① vision LLM detection (grounded boxes, 0–1000 norm coords)
                                          ② deep pass: 2 overlapping half-page crops (higher effective resolution)
                                          ③ self-verification: model inspects the BLACKENED page for leftovers
                                          ▼
                     padded/merged/unioned boxes ──burn──▶ image-only PDF (content destroyed, not hidden)
                                          ▼
                     ④ VERIFICATION LAYER (redacted vs. original):
                        · pixel diff — every intended box confirmed burned black,
                          unexpected content changes flagged (corruption/loss)
                        · LLM leak audit — fresh read of every REDACTED page;
                          anything personal still readable gets auto-fixed
                          (re-burned from the original, then re-audited)
```

**Speed:** pages and detection calls run in parallel (`--workers`, default 4),
API calls use JPEG transport and SSE streaming. An 11-page scanned protocol
processes in ≈3 min detection + ≈0.5 min verification (was ≈12 min sequential).

* **Detection** — any OpenAI-compatible vision endpoint (default: Qwen3-VL via
  OpenRouter) returns tight bounding boxes + a category per personal-data span
  (name, address, dob, phone, email, bank, id_number, signature).
* **Policy guards + referee** — deterministic German role-word/direct-dial guards
  and a per-span context-crop LLM verdict ("private person or company/official
  span?") keep companies and official persons visible. Guards fail closed:
  errors leave boxes ON (over-redaction, never a silent leak).
* **Deep pass** — the page is additionally analysed as two overlapping crops;
  boxes from all passes are **unioned** (a shifted duplicate is better than a gap).
* **Verification** — a second LLM call renders the *final blackened page* and
  reports anything personal still visible (incl. partially covered spans).
* **Numeric widening** — vision models systematically cut off the last digit
  groups of IBANs/IDs; numeric categories get a bounded width extension.
* **True redaction** — the output PDF is rebuilt from rasterized pages only.
  There is no text layer, no hidden objects: covered content cannot be recovered.

No OCR, no text extraction, no character recognition is used at any stage —
the model *looks* at the pixels, exactly like a human reviewer.

## Setup

```bash
pip install -r requirements.txt      # pymupdf, Pillow, requests
```

Endpoint & key configuration (env vars win over the project-root `.env`,
see `.env.example` in the repository root):

| Variable | Meaning | Default |
|---|---|---|
| `REDACT_API_URL` | OpenAI-compatible chat-completions endpoint | OpenRouter |
| `REDACT_MODEL` | model id at that endpoint | `qwen/qwen3-vl-235b-a22b-instruct` |
| `REDACT_API_KEY` (or `OPENROUTER_KEY`, `QWEN_KEY`, `OPENAI_API_KEY`) | API key | — |

To use your own Qwen server: `REDACT_API_URL=http://host:port/v1/chat/completions`.

## Usage

```bash
python redact_pdf.py scan.pdf                          # -> scan_redacted.pdf
python redact_pdf.py scan.pdf -o out/r.pdf --dpi 200    # higher res for small print
python redact_pdf.py scan.pdf --passes 2                # two verification rounds
python redact_pdf.py scan.pdf --no-deep                 # faster, less thorough
python redact_pdf.py scan.pdf --workers 6               # more parallelism
python redact_pdf.py scan.pdf --no-fix                  # audit reports leaks but doesn't auto-fix
python tools/verify.py original.pdf redacted.pdf        # standalone verification of an existing pair
```

Outputs per run:
* `<output>.pdf` — redacted, image-only PDF
* `<output>_review/page_NNN.png` — translucent overlay for human QA
* `<output>.report.json` — per-page boxes + categories (no text unless `--report-text`)

**Always spot-check the review PNGs before releasing a document.**

## Quality gate (deterministic)

`tests/test_grounding.py` measures detection quality against the *known* text
positions of the digitally generated sample (text extraction is used **only**
by this test as ground truth — never in production):

```bash
python tests/test_grounding.py
```

Result on the bundled sample (Qwen3-VL-235B): **16/16 PASS** — every PII span
(names incl. salutation & signature, private address, DOB, personnel number,
IBAN, private mobile & email, personal ID) covered ≥95% (11 of 12 at 100%),
while company name, address, phone and email remain 0% covered.

## Privacy & limitations

* Page images are sent to the OpenRouter model you configure — use a
  provider/model with a data policy that fits your compliance requirements
  (enterprise endpoints can be swapped in `redactor/config.py`).
* The tool redacts what a vision model recognises as personal data of natural
  persons; unusual handwriting, stamps or dense tables may need the review
  overlays. `--passes 2` adds a second verification round for critical docs.
* Encrypted PDFs are rejected; multi-page documents are fully supported.
