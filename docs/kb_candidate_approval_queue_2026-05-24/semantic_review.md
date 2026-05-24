# Semantic Review

Verdict: `PASS_FOR_QUEUE_SCHEMA_WITH_NOTES`

The queue is safe as an approval layer because it does not claim facts are approved, does not write to KB, and keeps `auto_import_allowed=false`.

## Checks

- Personal data: pass; no raw personal data included.
- Verbatim manager replies: pass; none included.
- Brand isolation: pass; each row has one active brand.
- Fact authority: pass with notes; candidate rows still need owner or primary-source approval before any KB import.
- P0 handling: pass with notes; D4 references `src/mango_mvp/channels/p0_recall_spec.py` but does not edit or stage `channels/`.

## Required Guard

Any confirmed semantic error found during approval must become a test, semantic gate, checklist item, or explicit manual-control reason.
