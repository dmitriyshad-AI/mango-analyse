# AMO writeback Stage40 + readback report, 2026-05-10

## Scope

This report records the post-backfill AMO CRM writeback stage after Stage15 v11 transcript quality hardening and CRM text quality gates.

Write target fields only:

- `Статус матчинга`
- `AI-приоритет`
- `AI-рекомендованный следующий шаг`
- `Последняя AI-сводка`
- `Авто история общения`

Protected fields remain out of scope and are not write targets:

- `Id Tallanto`
- `Филиал Tallanto`

## Source layer

- Input export: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage40_new_single_contact_ru.csv`
- Transcript quality gate: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`
- CRM quality gate for Stage40: `stable_runtime/crm_writeback_quality_gate_20260510_v12_stage40_new_single_contacts/summary.json`

Stage40 was built from Stage66 strict candidates after excluding:

- rows already written in earlier stage20 live writeback;
- AMO multi-contact matches;
- contact-id mismatch rows;
- Claude review-marker rows.

## Validation before live writes

### Stage40 gate

- Rows: 40
- Decision: 40 allow
- Blocking rows: 0
- Population recall: `passed_for_live=true`
- CRM text quality: `passed_for_live=true`

### Stage40 real-tunnel dry-run

- Run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T174739Z/`
- Rows: 40
- dry_run: 40
- skipped: 0
- failed: 0

## Live writeback stages

### Live stage A

- Run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T175140Z/`
- Input: first 20 rows of Stage40
- written: 20
- skipped: 0
- failed: 0

Readback gate:

- First attempt failed because the shared DB tunnel dropped before preflight.
- Successful retry: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T175140Z/readback_gate_v2/`
- evaluated rows: 20
- blocking rows: 0
- failed rows: 0
- passed: true

### Live stage B

- Input: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage20_remaining_after_live_readback_ru.csv`
- Quality gate: `stable_runtime/crm_writeback_quality_gate_20260510_v13_stage20_remaining_after_readback/summary.json`
- Gate rows: 20
- Decision: 20 allow
- Blocking rows: 0
- Real-tunnel dry-run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180304Z/`
- dry_run: 20
- skipped: 0
- failed: 0

Live writeback:

- Run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180418Z/`
- written: 20
- skipped: 0
- failed: 0

Readback gate:

- Folder: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180418Z/readback_gate/`
- evaluated rows: 20
- blocking rows: 0
- failed rows: 0
- passed: true

## Combined live status

Live runs considered here:

- `20260510T141007Z` earlier stage20: 20 written
- `20260510T175140Z` Stage40 live A: 20 written
- `20260510T180418Z` Stage40 live B: 20 written

Combined result:

- 60 unique AMO contact phones written across these three live stages.
- No phone overlap between the three live runs.
- Stage40 added 40 new live-written contacts beyond the earlier stage20.

## Audit note

A Claude audit pack was prepared at:

- `audits/_inbox/amo_stage40_live20_readback_remaining20_20260510_v1/`

Claude CLI could not be executed from the Codex sandbox because the sandbox could not access Claude home/auth files (`~/.claude.json`, lock directory) and returned auth error 401. The pack is kept for optional external/manual Claude audit.

## Current decision

The Stage40 live writeback is complete and readback-verified. Do not expand to larger live writeback until the remaining candidate classes are explicitly staged:

- multi-contact AMO matches;
- contact-id mismatch rows;
- Claude/manager review-marker rows;
- rows outside strict Stage40/Stage66 quality gates.

## Next recommended step

Build the next writeback candidate queue from post-backfill sources with explicit buckets:

1. `ready_single_contact_not_written` - safe candidates not in any previous written run.
2. `needs_manager_review_multi_contact` - multiple AMO exact contacts.
3. `blocked_contact_id_mismatch` - runtime/source contact-id mismatch.
4. `needs_text_quality_review` - review-marker or CRM text quality warnings.
5. `deferred_non_sales_or_service` - service/existing-client/no-content/out-of-domain rows.

Only bucket 1 can move to the next staged live writeback after quality gate, real-tunnel dry-run, and readback gate.

## Stage 1 production queue closure

After Stage40 live writeback and readback, the production queue was rebuilt with the dedicated queue builder:

- Script: `scripts/build_amo_writeback_queue.py`
- Queue folder: `stable_runtime/amo_writeback_queue_20260510_v2_production/`
- Source input: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_ru.csv`
- Manual review input: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage69_review_marker_rows_ru.csv`

Queue counts:

- `ready_single_contact_not_written`: 0
- `already_written`: 53
- `needs_manager_review_multi_contact`: 12
- `blocked_contact_id_mismatch`: 1
- `needs_text_quality_review`: 3
- `deferred_non_sales_or_service`: 0

Important policy correction implemented in the queue builder:

- `manual_review_input` and CRM text-quality blockers take priority over `already_written`.
- This prevents a row that was already written earlier but later became review-marked from being hidden as closed.

Current Stage 1 decision:

- No additional live writeback is allowed from the current strict AMO-ready layer because the safe bucket is empty.
- The remaining 16 rows are quarantined as review/block buckets: 12 multi-contact, 1 contact-id mismatch, 3 text-quality review.
- Next work is manager/manual resolution of those buckets or rebuilding a fresh strict input from newer data.

Audit pack prepared:

- `audits/_inbox/amo_writeback_stage1_completion_20260510_v1/`

## Additional production hardening after Stage 1 review

Implemented after subagent review:

- `write_amo_ready_contacts.py` now supports `--expected-written` and `--expected-dry-run` and fails the run if actual counts differ.
- Live/dry reports now store `preview_payload` and `payload_sha256` for every non-empty payload row, including live-written rows.
- `readback_amo_contact_writeback.py` now supports `--expected-evaluated`.
- Readback no longer passes on an empty/all-skipped selection.
- Readback now compares actual AMO field values against the payload recorded in the live writeback report when payload is available, and blocks `readback_value_mismatch`.

Post-hardening readback checks:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T175140Z/readback_gate_v3_expected20/summary.json` passed with `selected_source_rows=20`, `evaluated_rows=20`, `blocking_rows=0`, `failed_rows=0`.
- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180418Z/readback_gate_v2_expected20/summary.json` passed with `selected_source_rows=20`, `evaluated_rows=20`, `blocking_rows=0`, `failed_rows=0`.

Test status:

- `tests/test_amo_writeback_queue.py tests/test_amo_readback_gate.py tests/test_amo_writeback_guards.py`: 36 passed.
