"""Prompts for detection and verification passes (Qwen3-VL visual grounding,
coordinates normalized 0-1000).

POLICY: personal data of PRIVATE natural persons is redacted. Company data and
OFFICIAL PERSONS (notaries, court staff, employees of companies acting in
their business function) are exempt and stay visible."""

CATEGORIES = "name|address|dob|phone|email|bank|id_number|signature|other"

DETECT_PROMPT = """You are a GDPR redaction assistant. Find ALL visible personal data of PRIVATE natural persons in this scanned document page. Companies and official persons are EXEMPT — see KEEP rules.

REDACT (private natural persons only — e.g. customers, apartment owners, tenants, buyers, sellers, debtors, applicants, private contact persons):
- private person names — EVERY occurrence: address block, salutation, body text, letterhead, footer, signature block, participant/owner lists, tables
- private street / postal addresses of individuals
- dates of birth
- personnel / employee-ID / social-security / tax-ID numbers of private individuals
- private phone, mobile and fax numbers
- personal email addresses
- IBAN / BIC / bank account numbers of private individuals
- handwritten signatures of private persons

KEEP VISIBLE — do NOT box:
- ALL company data (companies are not a privacy concern): company names incl. legal form (GmbH, AG, KG, e.K., GbR, Ltd. ...), company addresses, switchboard phone/fax, company emails (info@, service@ ...), register numbers (HRB, HRB-Nr., Fn.), VAT/tax numbers of companies
- OFFICIAL PERSONS acting in their official or business capacity — their names, job titles and handwritten signatures stay visible:
  * notaries (Notar/Notarin) and notary-office staff
  * judges and court employees (Richter/in, Rechtspfleger/in, Urkundsbeamte ...)
  * employees of companies, courts and public authorities named in their business function — e.g. property managers, bank account managers, caseworkers. Cues: a function label under or next to the name (Notarin, Rechtspfleger, Kundenberaterin, Verwalterin), "i.A." / "i.V." / "gez." with a role, or the name sitting in the letterhead / signature block of the issuing company, court or authority
- pure document dates (letter date, effective date)
- all other business content

DECISION CUE for any person name — KEEP it as an OFFICIAL PERSON if ANY of these holds:
  * a function label stands directly BEFORE the name: "die Objektmanagerin Frau Neumann", "die Prokuristin Frau Lorenz", "die Rechtspflegerin Frau Siebert", "Ihren Betreuer Herrn Vogt", "der Sachbearbeiter Herr X", "der Notar Dr. Y"
  * a function label stands directly AFTER the name: "Dr. Falkenrath, Notarin", "Frau Neumann, Objektmanagerin", "gez. / i.A. / i.V. + name"
  * the name sits in the letterhead or signature block of the issuing company, court, notary's office or authority, or is the name of the notary office itself ("Notariat Dr. Falkenrath")
  * the person appears in a participant list ("Erschienen:") ONLY as a representative of a company (e.g. a bank's Prokuristin) — the COMPANY is the party, the person is its functionary
A name belongs to a PRIVATE party (-> REDACT) if the person is themselves the party in the matter: Verkäufer/Käufer, Eigentümer, Mieter, Kunde, Antragsteller, contact person in a private matter — even when the sentence mentions their role in the transaction ("als Verkäuferin: Frau Hoffmann"). The same person can appear in both roles: keep them where they act officially, redact them where they appear as a private party. When unsure, look at WHO the party to the document is: a company/court/notary acting through a person -> keep the person; an individual acting for themselves -> redact.

CRITICAL — BOX PRECISION: each box must tightly wrap ONLY the personal-data span itself, NOT the whole line or sentence.
LONG NUMBERS (IBAN, bank account, social-security, tax or personnel numbers): include the COMPLETE number — every digit group from the first to the last character, nothing cut off at either end. If a sentence reads "Für Rückfragen erreichen Sie Frau Dr. Katharina Schulze-Vohenstrauss", box only "Frau Dr. Katharina Schulze-Vohenstrauss". If a line mixes personal and non-personal text, output separate boxes, one per contiguous personal-data span.

Return STRICT JSON only, no markdown, no commentary:
{"boxes": [{"bbox_2d": [x1, y1, x2, y2], "category": "CATEGORY", "text": "<only the redacted span>"}]}
Coordinates: integers 0-1000 normalized to image width/height, x2 > x1, y2 > y1.
If nothing must be redacted, return {"boxes": []}."""

VERIFY_PROMPT = """This scanned document page has translucent red rectangles marking data that will be redacted.

Look ONLY for personal data of PRIVATE natural persons that is still VISIBLE and NOT covered by any red rectangle.
A span counts as MISSED even if only PART of it sticks out — e.g. an IBAN or phone number whose last digit groups extend beyond a red rectangle, or a name of which only the first words are covered. Report the FULL span, not just the uncovered part. (private person names incl. salutations, private addresses, dates of birth, personal ID numbers, private phone numbers, personal emails, IBAN/bank details, signatures of private persons).
Company data (names, addresses, contact data, register numbers) and OFFICIAL PERSONS acting in their official/business capacity — notaries, court staff, employees of companies/courts/authorities incl. their titles and signatures — are EXEMPT: they must remain visible and must NOT be reported.

Return STRICT JSON only:
{"missed": [{"bbox_2d": [x1, y1, x2, y2], "category": "CATEGORY", "text": "<the missed span>"}]}
Coordinates: integers 0-1000 normalized to image width/height.
If every personal-data span is already covered, return {"missed": []}."""

CLASSIFY_PROMPT = """You are the policy referee of a GDPR redaction tool. The image shows a scanned document page; below is a NUMBERED list of spans the detector marked for redaction. Decide for EACH span, using the full page as context, whether it is data of a PRIVATE person or not.

Only data of PRIVATE natural persons is redacted. A span stays VISIBLE (-> "keep") if it is:
- company data: company name/address/phone/fax/email, register or VAT numbers, bank or escrow accounts of companies, notaries, authorities (e.g. "Treuhandkonto des Notariats")
- an OFFICIAL PERSON in their function: notary and notary staff, judge or court employee, staff of a company/court/authority - role label before or after the name, letterhead/signature-block context, or representative capacity ("vertreten durch die Prokuristin ..."), incl. their names, titles and direct-dial numbers
- an institutional address of a court, authority, bank or notary office (the address block belonging to the institution, not to a private party)

A span stays REDACTED (-> "redact") if it is data of a PRIVATE party: customer, owner, tenant, buyer/seller, applicant, private contact person - their name, private address, date of birth, ID numbers, private phone/email, personal bank account.

Spans (number - category - text):
{spans}

Return STRICT JSON only - a verdict for EVERY number, "keep" = stays visible, "redact" = private-person data gets blacked out:
{"1": "keep", "2": "redact", "3": "redact", ...}
If every span belongs to a private person, every verdict is "redact"."""


SPAN_VERDICT_PROMPT = """A scanned document fragment is shown; ONE span is marked with a red rectangle. Decide if that span is personal data of a PRIVATE person.

"redact" (private party: customer, owner, tenant, buyer/seller, applicant, private contact person - name, private address, date of birth, ID number, private phone/email, personal bank account) or
"keep" (anything else: company data incl. addresses/phones/emails/register numbers; bank or escrow accounts of companies/notaries/authorities; institutional addresses of courts/banks/authorities; OFFICIAL PERSONS in function - notary and staff, judges/court employees, company/court/authority staff - recognizable by a role label next to the name (Notarin, Rechtspflegerin, Prokuristin, Objektmanagerin, Sachbearbeiter ...), letterhead/signature-block context, or "i.A."/"i.V."/"gez." markers).

IMPORTANT — read the CONTEXT around the marked span before answering. A person name is a company/court/notary FUNCTIONARY (-> keep) when the surrounding text names their role, even if the role word is not inside the rectangle: "erreichen Sie unsere Objektmanagerin Frau Neumann" -> Frau Neumann is a functionary -> keep. "die Prokuristin Frau Lorenz" -> keep. "Rechtspflegerin Frau Siebert" -> keep. "Ihren Betreuer Herrn Vogt" (company contact person) -> keep. "als Verkäuferin: Frau Hoffmann" -> private party -> redact.
A bank/escrow IBAN belonging to a company, notary office or authority (e.g. "Treuhandkonto des Notariats") -> keep. A personal bank account of a private party -> redact.
RULE: if a business/court/notary ROLE NOUN appears anywhere in the same LINE as the marked person name (before or after it, outside the rectangle too) - Objektmanagerin, Betreuer, Sachbearbeiter, Kundenberater, Prokurist/in, Notar/in, Rechtspfleger/in, Richter/in, Verwalter/in, Geschäftsführer/in - the marked person is a functionary: verdict "keep". Only when no role noun occurs in the line and the person is a party to the matter (owner, tenant, buyer, seller, customer) is the verdict "redact".
Use the surrounding context in the image to decide.
Return STRICT JSON only: {"verdict": "redact" | "keep", "why": "<max 8 words>"}"""