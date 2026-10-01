"""Prompts for detection and verification passes (Qwen3-VL visual grounding,
coordinates normalized 0-1000)."""

CATEGORIES = "name|address|dob|phone|email|bank|id_number|signature|other"

DETECT_PROMPT = """You are a GDPR redaction assistant. Find ALL visible personal data of NATURAL PERSONS in this scanned document page.

REDACT (natural persons only):
- person names — EVERY occurrence: address block, salutation, body text, letterhead, footer, signature block
- private street / postal addresses of individuals
- dates of birth
- personnel / employee / social-security / tax ID numbers
- private phone, mobile and fax numbers
- personal email addresses
- IBAN / BIC / bank account numbers
- handwritten signatures of individuals

KEEP VISIBLE — do NOT box:
- company names incl. legal form (GmbH, AG, KG, e.K., Ltd. ...)
- company addresses and company contact data (switchboard phone, info@ emails)
- pure document dates (letter date, effective date)
- all other business content

CRITICAL — BOX PRECISION: each box must tightly wrap ONLY the personal-data span itself, NOT the whole line or sentence.
LONG NUMBERS (IBAN, bank account, social-security, tax or personnel numbers): include the COMPLETE number — every digit group from the first to the last character, nothing cut off at either end. If a sentence reads "Für Rückfragen erreichen Sie Frau Dr. Katharina Schulze-Vohenstrauss", box only "Frau Dr. Katharina Schulze-Vohenstrauss". If a line mixes personal and non-personal text, output separate boxes, one per contiguous personal-data span.

Return STRICT JSON only, no markdown, no commentary:
{"boxes": [{"bbox_2d": [x1, y1, x2, y2], "category": "CATEGORY", "text": "<only the redacted span>"}]}
Coordinates: integers 0-1000 normalized to image width/height, x2 > x1, y2 > y1.
If nothing must be redacted, return {"boxes": []}."""

VERIFY_PROMPT = """This scanned document page has translucent red rectangles marking data that will be redacted.

Look ONLY for personal data of NATURAL PERSONS that is still VISIBLE and NOT covered by any red rectangle.
A span counts as MISSED even if only PART of it sticks out — e.g. an IBAN or phone number whose last digit groups extend beyond a red rectangle, or a name of which only the first words are covered. Report the FULL span, not just the uncovered part. (person names incl. salutations and signatures, private addresses, dates of birth, personal ID numbers, private phone numbers, personal emails, IBAN/bank details). Company names and company contact data must remain visible and must NOT be reported.

Return STRICT JSON only:
{"missed": [{"bbox_2d": [x1, y1, x2, y2], "category": "CATEGORY", "text": "<the missed span>"}]}
Coordinates: integers 0-1000 normalized to image width/height.
If every personal-data span is already covered, return {"missed": []}."""
