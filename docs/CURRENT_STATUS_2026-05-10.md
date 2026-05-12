# Current Status, 2026-05-10

Статус: draft для handoff/операционного ориентира.

Scope этого документа: зафиксировать, какой слой сейчас считается актуальным, чем закончился AMO writeback Stage 1, и какие данные должны оставаться в quarantine/manual-review до отдельного решения.

Этот документ не авторизует live writeback, физическое удаление, архивирование или изменение `stable_runtime`.

## Executive Summary

- Актуальный canonical/writeback source-of-truth на 2026-05-10: `stable_runtime/CANONICAL_EXPORT.txt` указывает на `sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`.
- Под ним лежит post-backfill canonical DB: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`.
- Stage15 v11 frozen gate passed; CRM writeback readiness passed. Bot autonomous production отдельно не готов из-за over-sanitization review queue, но это не блокирует internal CRM writeback.
- AMO writeback Stage 1 завершен и безопасно остановлен: safe bucket `ready_single_contact_not_written=0`.
- Из текущего strict AMO-ready слоя нельзя продолжать live writeback без нового staged scope: оставшиеся строки находятся в quarantine/review buckets.

## Current Source Of Truth

### Canonical DB

Актуальный canonical слой после quality backfill:

- DB: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`
- Summary: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/summary.json`
- Source audio: `64 867`
- Excluded manager-manager/no-ASR: `35`
- Actionable source audio: `64 832`
- ASR done actionable: `64 832`
- Full R+A actionable: `64 832`
- Missing ASR actionable: `0`
- Missing full R+A actionable: `0`
- Duplicate source names with candidates: `35 604`
- Validation: passed

Interpretation: this is the current canonical call layer for downstream rebuilds. Older `canonical_master_20260509_*` layers remain useful audit/provenance evidence, but are not the current post-backfill source for CRM writeback.

### Phone-Chain / Insight Layer

Current phone-chain context:

- `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/client_chains.csv`
- Summary: `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/summary.json`
- Terminal analyzed calls: `64 832`
- Calls with phone: `63 788`
- Unique client phones: `15 924`
- Client chains with contentful calls: `13 477`
- Contentful calls: `46 153`
- Non-conversation calls: `18 679`

Interpretation: this is the current contact/chain layer used to build CRM-oriented exports. It is read context, not a direct live writeback artifact.

### Stage15 / Bot Boundary

Current Stage15 gate:

- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`
- `passed=true`
- `readiness.crm_quality_writeback_ready=true`
- `readiness.bot_allowlist_export_ready=true`
- `readiness.bot_autonomous_production_ready=false`

Interpretation: CRM internal writeback can use CRM-bound text after the staged gates. Bot/autonomous use remains stricter and still requires over-sanitization queue review before production autonomy.

### Current CRM/AMO Export Pointer

Current pointer:

- `stable_runtime/CANONICAL_EXPORT.txt` -> `sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`

Current strict export:

- Root: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/`
- Summary: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/summary.json`
- `master_calls_rows=64 832`
- `master_contacts_rows=15 923`
- `amo_export_ready_rows=69`
- `manual_review_rows=12 297`
- `contentful_calls=43 514`
- `non_conversation_calls=21 318`
- `stage15_passed=true`
- `crm_quality_writeback_ready=true`

Current strict CRM quality gate:

- `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/summary.json`
- Rows: `69`
- Decision: `allow=69`
- Blocking rows: `0`
- CRM text quality: `passed_for_live=true`

Interpretation: the strict v5 export is the current evaluated CRM writeback candidate layer. It is not a blanket live-write authorization; it feeds staged dry-run/live/readback gates and the production queue.

## AMO Writeback Stage 1 Result

Target fields for the staged writeback were limited to:

- `Статус матчинга`
- `AI-приоритет`
- `AI-рекомендованный следующий шаг`
- `Последняя AI-сводка`
- `Авто история общения`

Protected fields remain out of write scope:

- `Id Tallanto`
- `Филиал Tallanto`

Live/readback evidence:

- Earlier Stage20 live run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T141007Z/`, `20` written.
- Stage40 live A: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T175140Z/`, `20` written, readback passed.
- Stage40 live B: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180418Z/`, `20` written, readback passed.
- Combined Stage40 report records `60` unique AMO contact phones written across those three staged live runs, with no phone overlap between the three runs.

Stage1 production queue:

- Queue root: `stable_runtime/amo_writeback_queue_20260510_v2_production/`
- Queue source: current strict `amo_export_ready_ru.csv`, `69` rows
- `ready_single_contact_not_written=0`
- `already_written=53`
- `needs_manager_review_multi_contact=12`
- `blocked_contact_id_mismatch=1`
- `needs_text_quality_review=3`
- `deferred_non_sales_or_service=0`

Important interpretation:

- Stage1 is closed because there are no safe not-yet-written single-contact rows left in the current strict layer.
- The queue intentionally prioritizes `manual_review_input` and CRM text-quality review over `already_written`. This prevents a row that was written earlier but later became review-marked from being hidden as closed.
- No additional live writeback should be run from the current strict layer until a new source layer or manually resolved bucket produces a fresh non-empty safe bucket and passes all gates.

Existing test evidence from the Stage1 audit pack:

- `tests/test_amo_writeback_queue.py`
- `tests/test_amo_readback_gate.py`
- `tests/test_amo_writeback_guards.py`
- Reported status: `36 passed`

## What Is Current

Treat these as current operational anchors:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/`
- `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/`
- `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/`
- `stable_runtime/amo_writeback_queue_20260510_v2_production/`
- Readback-verified live run folders: `20260510T141007Z`, `20260510T175140Z`, `20260510T180418Z` under `stable_runtime/amocrm_runtime/contact_writebacks/`.
- Docs: `docs/AMO_WRITEBACK_STAGE40_READBACK_REPORT_2026-05-10.md` and `docs/AMO_WRITEBACK_PRODUCTION_LOOP_2026-05-10.md`.

Treat these as historical/superseded but keepable evidence:

- `sales_master_export_20260510_after_quality_backfill_v1` through `v4`.
- Earlier CRM quality gate roots before `v10_crm_text_quality_strict`, unless a specific audit cites them.
- `canonical_master_20260509_*` roots: useful for provenance of the first canonical build, not current post-backfill CRM source.
- Old April `sales_master_export_*` layers: not current for CRM writeback.

## What Is Quarantined

### Stage1 AMO Queue Quarantine

From `amo_writeback_queue_20260510_v2_production`:

- `needs_manager_review_multi_contact=12`: runtime AMO dry-run found multiple exact contacts; requires manual/manager contact choice before any future write.
- `blocked_contact_id_mismatch=1`: source AMO contact id and runtime-resolved AMO contact id disagree; must stay blocked until linkage drift is resolved.
- `needs_text_quality_review=3`: review-marker/manual-review input takes priority over already-written status; requires manual text review or fresh strict rebuild.

### Strict Export Manual Review

Current strict export manual review queue:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/review_queues/manual_review_contacts_current.csv`
- Rows: `12 297`

Main reason buckets:

- `6 845`: contentful calls with manual review required.
- `3 400`: no AMO contact ID in post-backfill phone chain; live lookup/create policy required.
- `1 469`: service_call / existing-client context plus missing AMO contact ID.
- `397`: existing_client_progress context plus missing AMO contact ID.
- `107`: technical_call context plus missing AMO contact ID.
- `33`: passive or closing next step; manual review before AMO writeback.
- `28`: service_call / existing-client context, not a new sales lead.
- `9`: multiple AMO contact IDs; manual contact selection required.
- `5`: existing_client_progress / existing-client context, not a new sales lead.
- `2`: CRM text quality `closure_next_step_requires_downgrade`.
- `1`: technical_call / existing-client context, not a new sales lead.
- `1`: CRM text ends with ellipsis / probable truncation.

These rows are not safe live-write candidates. They may become eligible only after explicit manual resolution, fresh export, quality gate, real-tunnel dry-run, live approval, and readback.

### Bot/Autonomous Quarantine

Bot/autonomous production remains blocked by Stage15 readiness:

- Blocker: `over_sanitization_queue_requires_rop_review_before_autonomous_bot`

This is separate from internal CRM value. CRM-bound fields may intentionally contain operational sales context, prices, product names, branches, and manager-facing history. That content must not be copied into the bot-safe/autonomous allowlist without Stage15/bot-specific gates.

## Gates Before Any Next Live Stage

Before any next staged AMO writeback, require all of the following:

- Current pointer still resolves to the intended strict export.
- current export `summary.json` references `canonical_master_20260510_after_quality_backfill_v1` and `insight_readiness_report_after_quality_backfill_20260510_v1`.
- Stage15 summary is `passed=true` and `crm_quality_writeback_ready=true`.
- CRM writeback quality gate is `passed=true`, `blocking_rows=0`, and input path matches the candidate CSV.
- Candidate queue has a non-empty `ready_single_contact_not_written` bucket.
- Real-tunnel dry-run returns zero failed/skipped rows for the intended staged scope.
- Live write uses explicit confirmation and expected-count guards.
- Post-writeback readback evaluates the expected row count and blocks mismatches.

Stage50/full rollout is not authorized by this status document.
