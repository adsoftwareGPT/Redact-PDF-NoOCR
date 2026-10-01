# pdf-redaction

One project, two faces — everything for LLM-based PDF redaction (no OCR):

```
pdf-redaction/
├── engine/   ← ex "no-ocr-redaction": the CLI engine
│   ├── redact_pdf.py        CLI: python3 redact_pdf.py scan.pdf
│   ├── redactor/            detection · burn · verify (shared package)
│   ├── tests/  samples/  tools/
│   └── requirements.txt     pymupdf, Pillow, requests
└── web/      ← ex "pdf-redact-web": the browser UI
    ├── server.py            FastAPI — python3 web/server.py → http://localhost:8799
    ├── static/              UI (upload · toggle boxes · draw custom · export)
    └── jobs/                persisted jobs (survive restarts, deep-link #jobid)
```

The web UI imports the engine directly (`../engine` on sys.path) — one codebase,
one API key (`C:\agent_linux\.env`, `OPENROUTER_KEY`), one model (Qwen3-VL).

## Quick start
```bash
cd Desktop/pdf-redaction
python3 web/server.py          # UI at http://localhost:8799
# or CLI:
python3 engine/redact_pdf.py "doc.pdf"   # writes doc_redacted.pdf + QA overlays + report
```

Redacted exports land in `Desktop\redacted\`.
