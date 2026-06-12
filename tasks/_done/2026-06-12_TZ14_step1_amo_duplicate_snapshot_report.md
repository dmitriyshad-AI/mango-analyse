# TZ-14 Step 1 AMO Duplicate Snapshot Report

Date: 2026-06-12

## Scope

Executed TZ-14 Step 1: fresh read-only AMO snapshot through `foton-crm-readonly` MCP and duplicate/common-phone/multi-child metrics.

No AMO/Tallanto/CRM write was executed.

## Commit Plan

Step 1 is prepared as a separate commit:

`TZ14 step1 AMO duplicate snapshot`

## Live Read Parameters

- MCP tool: `amo_api_get`
- AMO paths: `leads/pipelines`, `contacts`, `leads`
- Page limit: 10
- Pause: 1.05 seconds
- Transport: MCP JSON-RPC over HTTPS via `curl`
- Checkpointing: enabled

## Counts

- Contacts seen: 13,895
- Leads seen: 8,004
- Contact pages: 1,390
- Lead pages: 801
- Contacts with phone: 13,275
- Contacts without phone: 620
- Unique phones: 13,654
- Phones with 2+ contacts: 689
- Duplicate groups total: 29
- Duplicate contact rows total: 60
- Live duplicate groups: 24
- Live duplicate contact rows: 50
- Weekly duplicate groups: 3
- Weekly new duplicate contacts: 4
- Possible common phone groups: 113
- Multi-child family groups: 537
- Ambiguous missing-parent groups: 6

## Output Artifacts

Raw CRM artifacts are intentionally outside git:

`product_data/customer_profiles/tz14_amo_step1_full_20260612/`

Files:
- `amo_contacts_raw.jsonl`
- `amo_leads_raw.jsonl`
- `amo_pipelines.json`
- `amo_step1_snapshot.sqlite`
- `duplicate_candidates.csv`
- `common_phone_review.csv`
- `multi_child_families.csv`
- `ambiguous_phone_review.csv`
- `summary.json`

Git ignore check passed for this path through `.gitignore: product_data/customer_profiles/`.

## Tests And Checks

- Full pytest: `3047 passed, 2 skipped, 1 warning`
- Targeted Step 1 pytest: `6 passed`
- SQLite `PRAGMA quick_check`: `ok`
- Idempotent rerun: CSV checksums unchanged
- Read-only grep: Step 1 uses MCP JSON-RPC POST transport only; no AMO/Tallanto write tool or write endpoint is called by the Step 1 code

Test logs:
- `audits/_inbox/tz14_step1_amo_20260612/full_pytest_output.txt`
- `audits/_inbox/tz14_step1_amo_20260612/targeted_pytest_output.txt`

Audit pack:

`audits/_inbox/tz14_step1_amo_20260612/`

## NEG

- Different children on the same phone are not silently merged.
- Shared/common phone cases are separated from duplicate candidates.
- Missing-parent ambiguous cases are separated from duplicate candidates.
- Repeated build with completed checkpoints is idempotent for CSV outputs.
- Output inside the repo is refused outside `product_data/customer_profiles/`.
- 429 retry and socket timeout retry are covered by tests.
- No live write flags or write methods are part of Step 1.

## LLM Calls

`llm_calls_total`: 0 for Step 1 scripts.

## Notes

The final full run used `limit=10` because larger connector responses previously caused timeout or truncation behavior. This increased runtime but avoided partial output.

`page_limit=10` was an operator safety choice for connector response size, not an AMO API cap; confirmation: the completed run fetched 1,390 contact pages and 801 lead pages without partial-output rejection.

The `live_candidate` marker is a review hint only. It means at least one record in the duplicate group has an active AMO lead or Tallanto link; it is not permission to merge records.
