# TZ-14 Step 4 Tallanto Live Card Report

Date: 2026-06-12

## Scope

Implemented Step 4: read-only `live_card` v1 inside Tallanto context.

No Tallanto/AMO/CRM write was executed. Bot prompt consumption was not changed.

## Commit

Prepared as separate Step 4 commit:

`TZ14 step4 Tallanto live card`

## Contract

Existing response keys are preserved:
- `enabled`
- `status`
- `matched_via`
- `contacts_found`
- `contexts`

New key:
- `live_card`

`live_card` contains:
- `payments`
- `balance`
- `schedule`
- `enrollment`
- `ttl_seconds`
- `skipped`
- `_provenance`

## Stop Rules

- no Tallanto contact: no card
- multiple Tallanto contacts: no card
- `filial=shd`: no card, not Foton
- brand mismatch with `active_brand`: no card
- inactive classes: filtered out
- teacher names: not copied into card
- `remaining_seats=10000`: not shown as a real seat count

## Live Schema Probe

Stored at:

`audits/_inbox/tz14_step4_tallanto_live_card_20260612/live_field_probe.txt`

Confirmed:
- `most_class`: `status`, `filial`, `date_start`, `audience`, `number_seats`, `remaining_seats`
- `most_abonements`: `contact_id`, `num_visit_left`, `filial`
- `ClassContactsRelationship`: `contact_id`, `most_class_id`
- `status=active` filter returns active rows
- `status=notactive` filter returns notactive rows

No names, phones, emails, or raw records were written to the report.

## Tests And Checks

- Full pytest: `3067 passed, 2 skipped, 1 warning`
- Targeted Tallanto tests: `12 passed`
- `py_compile`: passed
- Read-only grep: no Tallanto write methods or AMO write helpers in Step 4 files

Audit pack:

`audits/_inbox/tz14_step4_tallanto_live_card_20260612/`

## LLM Calls

`llm_calls_total`: 0 for Step 4 scripts/code.

## Status

`formal_pass`: yes.

`semantic_pass`: `PASS_WITH_NOTES` for read-only source layer.

This is not a bot prompt rollout.
