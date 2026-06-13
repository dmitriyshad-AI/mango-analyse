Semantic review for E1-E5

E1:
- Phone keys are now derived from one canonical function.
- Output contracts intentionally remain different where callers already depended on them: plus-format for CRM/timeline/mail, digit-only for insight chains.
- Short junk such as "123" is rejected instead of becoming a match key.

E2:
- Default OFF keeps old profile DB schema and full-scan lookup.
- ON creates an indexed normalized phone column and preserves summary lookup results in tests.

E3:
- Explicit analysis_date="2026-05-13" remains byte-stable for old fixtures.
- New default no longer silently writes May 13, 2026 into fresh runs.

E4:
- New Analyze metadata is additive; existing readers of analysis_meta/quality_flags continue to see the old structure.

E5:
- Candidate scripts were only listed with evidence; no deletion or rename was done.

Residual semantic risk:
- E2 production DB migration/backfill is not run here. Existing DBs without primary_phone_norm intentionally fall back to old behavior.
- E4 affects only future Analyze writes; old analysis_json rows are not rewritten.
