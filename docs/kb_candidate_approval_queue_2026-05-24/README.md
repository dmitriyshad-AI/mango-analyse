# KB Candidate Approval Queue 2026-05-24

Purpose: safe queue for future KB enrichment from email, Telegram, calls, OCR and current KB reviews.

This package is not a KB release and does not import facts. It only defines candidate rows and approval states.

## Safety Rules

- No personal data: no phone, email, full name, username, AMO/Tallanto id, raw filename, raw message link.
- No verbatim manager replies.
- One candidate row = one active brand: `foton` or `unpk`.
- Historical channels are evidence only until human approval.
- `auto_import_allowed=false` for every row.
- `approved_for_kb_import=false` for every initial row.
- P0 checks must reference `src/mango_mvp/channels/p0_recall_spec.py`; D4 does not create its own P0 list.

## Files

- `STATUS_RULES.md` - allowed statuses and transitions.
- `approval_queue.csv` - human-readable queue.
- `approval_queue.jsonl` - machine-readable copy.
- `approval_decisions_template.csv` - template for Dmitry/ROP decisions.
- `blocked_or_rejected_items.csv` - unsafe or cross-brand candidates.
- `source_refs_masked.csv` - masked references only.
- `brand_isolation_report.csv` - single-brand check.
- `pii_redaction_report.md` - personal-data safety note.
- `semantic_review.md` - meaning and safety review.
- `safe_creation_notes.md` - creation notes.
