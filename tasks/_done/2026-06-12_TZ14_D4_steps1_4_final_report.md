# TZ-14 D4 Steps 1-4 Final Report

Date: 2026-06-12

## Scope

Executed TZ-14 D4 steps 1-4 locally on branch `codex/tz14-d4-crm-amo-tallanto`.

Step 0 dependency was checked before AMO work. Local report:

`tasks/_done/2026-06-12_TZ14_step0_amo_token_report.md`

Claude control env file was copied to:

`/Users/dmitrijfabarisov/Claude Projects/Foton/foton_crm_readonly_mcp_connector.env`

The file has mode `600`. Secret contents were not printed or committed.

Step 5 was not executed.

No AMO/Tallanto/CRM write was executed. No bot prompt rollout was executed.

## Commits

- `1a8f149f` - `TZ14 step1 AMO duplicate snapshot`
- `c7c52bb8` - `TZ14 step2 AMO new lead scan`
- `1d3ebd12` - `TZ14 step3A contact card dry run`
- `a6162d7e` - `TZ14 step4 Tallanto live card`

This final report and the Step 0 local report are committed separately after the implementation commits.

## Step 1: AMO Duplicate Snapshot

Raw artifacts are outside git:

`product_data/customer_profiles/tz14_amo_step1_full_20260612/`

Counters:

- contacts seen: 13,895
- leads seen: 8,004
- contact pages: 1,390
- lead pages: 801
- contacts with phone: 13,275
- contacts without phone: 620
- unique phones: 13,654
- phones with 2+ contacts: 689
- duplicate groups total: 29
- live duplicate groups: 24
- weekly duplicate groups: 3
- possible common phone groups: 113
- multi-child family groups: 537
- ambiguous missing-parent groups: 6

`page_limit=10` was an operator safety choice for connector response size, not an AMO API cap; confirmation: the completed run fetched 1,390 contact pages and 801 lead pages without partial-output rejection.

Audit pack:

`audits/_inbox/tz14_step1_amo_20260612/`

## Step 2: AMO New Lead Scan

Raw dry-run artifacts are outside git:

`product_data/customer_profiles/tz14_amo_step2_pilot_24h_v2_20260612/`

Review sample:

`product_data/customer_profiles/tz14_amo_step2_review_sample_20260612/`

Live microprobe since `2026-06-11T00:00:00+00:00`:

- leads seen: 10
- contacts fetched: 9
- family note drafts: 3
- known-family notes: 2
- common-phone notes: 1
- leads without profile: 7
- callback task drafts: 0
- rerun family note drafts: 0
- rerun already-journaled draft skips: 3

Important correction: missing-profile skips are not permanently journal-blocked, so later profile matches can still generate drafts.

Audit pack:

`audits/_inbox/tz14_step2_amo_scan_20260612/`

## Step 3A: AMO Contact Card Dry Run

Live field check:

- field: `ИИ: профиль клиента`
- field id: `2363933`
- field type: `textarea`
- API-only: `false`
- status: `ok`

Stage A package for Dmitry review:

`product_data/customer_profiles/tz14_amo_step3_stage_a_20260612/`

Counters:

- requested families: 20
- selected families: 20
- card rows: 20
- finding rows: 0
- skipped missing phone: 1
- skipped no AMO contact: 2
- skipped no live contact: 5
- skipped phone with 2+ profiles: 0

Quality scan:

- raw phone rows: 0
- mixed brand rows: 0
- blocked rows: 0

Audit pack:

`audits/_inbox/tz14_step3_contact_cards_20260612/`

Stage B live writeback is not included.

## Step 4: Tallanto Live Card

Implemented read-only `live_card` v1 inside Tallanto context.

Existing response keys are preserved:

- `enabled`
- `status`
- `matched_via`
- `contacts_found`
- `contexts`

New key:

- `live_card`

`live_card` includes payments, balance, schedule, enrollment, TTL, skipped reasons, and provenance.

Stop rules:

- no Tallanto contact: no card
- multiple Tallanto contacts: no card
- `filial=shd`: no card, not Foton
- brand mismatch with `active_brand`: no card
- inactive classes: filtered out
- teacher names: not copied into card
- `remaining_seats=10000`: not shown as a real seat count

Live schema probe:

`audits/_inbox/tz14_step4_tallanto_live_card_20260612/live_field_probe.txt`

Audit pack:

`audits/_inbox/tz14_step4_tallanto_live_card_20260612/`

This is not a bot prompt rollout.

## Tests

Latest full pytest after Step 4:

`3067 passed, 2 skipped, 1 warning`

Targeted checks:

- Step 1: `6 passed`
- Step 1 + Step 2 + PII ignore: `15 passed`
- Step 3: `4 passed`
- Tallanto Step 4: `12 passed`

Step 3 full pytest was rerun because the first full run had unrelated/flaky failures that passed separately. Final full rerun passed.

## NEG Coverage

- AMO/Tallanto live write flags fail closed or are not implemented.
- Shared/common phone cases are separated from duplicate candidates.
- Different children are not silently merged.
- Missing-profile skips are not permanently blocked.
- Callback JSONL malformed line is skipped with a counter.
- Callback negative phrase is not treated as a callback request.
- Repeated dry-run scan is idempotent.
- Contact cards block raw phone-like values.
- Contact cards block mixed brand markers.
- Wrong AMO field type is reported and not written.
- Tallanto `shd` filial is skipped.
- Tallanto multiple contacts produce no card.
- Inactive classes are filtered.
- Teacher names are not copied into live cards.

## LLM Calls

`llm_calls_total`: 0 for all TZ-14 Step 1-4 scripts/code paths.

## Git And Data Hygiene

Committed implementation files contain no raw CRM/Tallanto exports, no secrets, and no PII fixtures.

Raw artifacts are under ignored `product_data/customer_profiles/`.

The scoped TZ-14 implementation files are committed. The global worktree is not fully clean because these unrelated files were already modified by another track and were not touched:

- `CLAUDE.md`
- `scripts/run_amo_wappi_draft_loop.py`
- `src/mango_mvp/integrations/draft_loop.py`
- `tests/test_draft_loop.py`
- `tests/test_run_amo_wappi_draft_loop.py`

## Decision Boundaries

Dmitry reviews the 20-family Step 3A package before any Stage B decision.

No CRM writeback, no Tallanto writeback, no profile rebuild, and no bot prompt consumption change were started by this work.
