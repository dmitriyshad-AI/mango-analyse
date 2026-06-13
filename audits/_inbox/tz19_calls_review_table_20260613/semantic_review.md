# TZ-19 calls review table semantic review

Verdict: `PASS_WITH_NOTES`

## Artifact and audience

Artifact: internal Excel workbook for Dmitry's manual review of Analyse v7 call summaries.

Audience: owner/internal analyst. Not customer-facing, not a CRM write payload.

## What passed

- The workbook is explicitly for manual review before any decision about CRM/student-card writes.
- The workbook keeps summaries columnar instead of exporting the full `history_summary` blob.
- Raw phone numbers and emails were not found by regex scan in generated workbook cells.
- The report in git contains only counts and paths, no row-level personal data.
- Blacklist rows are not imported or changed.

## Non-blocking risks

- Some free-text summaries may still contain non-regex personal hints if the model wrote them without matching structured fields. The workbook is therefore kept outside git.
- Brand detection is best-effort from text/tags and is often `unknown`; it should not be treated as authoritative.
- `blacklist-77` sheet is empty because blacklist calls are not part of the 22,679 v7 scope. This is correct for this ТЗ but should be called out to the reviewer.

## Missing checks

- No manual semantic grading of the 22,679 rows was performed; the workbook is the input for that manual review.
- No decision was made about writing any field into CRM/Tallanto/student cards.

## Required next action

Dmitry reviews the workbook and decides which fields are safe candidates for future CRM/student-card text. That future write/design stage must be a separate ТЗ.
