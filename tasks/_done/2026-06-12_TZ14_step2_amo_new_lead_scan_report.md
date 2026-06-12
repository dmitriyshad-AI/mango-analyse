# TZ-14 Step 2 AMO New Lead Scan Report

Date: 2026-06-12

## Scope

Implemented Step 2 as a read-only/dry-run scanner:
- new AMO leads through `foton-crm-readonly`;
- family note drafts for known phones;
- neutral shared-phone note drafts;
- callback task drafts from JSONL input;
- local idempotency journal.

No AMO/Tallanto/CRM write was executed.

## Commit

Prepared as separate Step 2 commit:

`TZ14 step2 AMO new lead scan`

## Live Microprobe

Clean dry-run folder:

`product_data/customer_profiles/tz14_amo_step2_pilot_24h_v2_20260612/`

Separate review sample with draft CSV rows preserved:

`product_data/customer_profiles/tz14_amo_step2_review_sample_20260612/`

Initial probe, since `2026-06-11T00:00:00+00:00`:
- leads seen: 10
- contacts fetched: 9
- family note drafts: 3
- known-family notes: 2
- common-phone notes: 1
- leads without profile: 7
- write_crm: false

Rerun on same folder:
- family note drafts: 0
- already-journaled draft skips: 3
- leads without profile: 7

Review sample CSV rows:
- `family_note_drafts.csv`: 3
- `callback_task_drafts.csv`: 0
- `skipped.csv`: 7

Important fix: missing-profile skips are not journal-blocked, so a later profile match can still generate a draft.

## Tests And Checks

- Full pytest: `3053 passed, 2 skipped, 1 warning`
- Targeted Step 1 + Step 2 + PII ignore tests: `15 passed`
- Step 2 synthetic tests: `6 passed`
- SQLite journal `PRAGMA quick_check`: `ok`
- Raw dry-run paths are ignored by git through `product_data/customer_profiles/`
- Secret grep: no secrets in new Step 2 files

Audit pack:

`audits/_inbox/tz14_step2_amo_scan_20260612/`

## NEG

- Live write flags fail closed with `PermissionError`.
- Repeated scan does not duplicate note/task drafts.
- Leads without phone are skipped.
- Leads without current profile are skipped but not permanently blocked.
- Shared/common phone note is neutral and does not include names or children.
- Callback request JSONL tolerates malformed lines and counts them.
- Negative phrase like “я сам перезвоню” is not treated as a callback request.
- Callback dedupe key is `(chat_id, date, intent=callback)`.
- Callback task text does not duplicate the phone.
- Output under the repo is restricted to ignored `product_data/customer_profiles/`.

## LLM Calls

`llm_calls_total`: 0 for Step 2 scripts.

## Status

`formal_pass`: yes.

`semantic_pass`: `PASS_WITH_NOTES` for dry-run review only.

This is not a live CRM write implementation and not permission to create AMO notes/tasks.
