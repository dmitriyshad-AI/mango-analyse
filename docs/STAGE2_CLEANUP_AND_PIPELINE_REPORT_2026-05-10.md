# Stage2 Cleanup And Pipeline Report, 2026-05-10

Статус: draft.

Scope: зафиксировать Stage2 cleanup/pipeline позицию после canonical master, quality backfill и AMO writeback Stage1 closure. Документ описывает, что считать актуальным, что держать как evidence, и что отправлять в quarantine/manual-review.

Этот документ не выполняет cleanup, не удаляет данные, не архивирует runtime и не запускает live writeback.

## Stage2 Position

Stage2 cleanup не должен начинаться с удаления. Текущая безопасная позиция:

1. Canonical layer после quality backfill уже собран и validated.
2. Downstream CRM/writeback слой переведен на strict v5 export.
3. AMO writeback Stage1 завершен: safe bucket пустой.
4. Следующий cleanup шаг - классификация active/superseded/quarantine, а не физический delete.

## Current Pipeline Map

```text
source audio 2025-01..2026-05
  -> canonical_master_20260510_after_quality_backfill_v1
  -> insight_readiness_report_after_quality_backfill_20260510_v1
  -> transcript_quality_stage15_export_gate_20260510_v11_frozen_gate
  -> sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict
  -> crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict
  -> staged AMO dry-run/live/readback
  -> amo_writeback_queue_20260510_v2_production
```

Operational rule: downstream work should read from the latest validated layer in this chain, not from old monthly batch DBs or earlier `sales_master_export_*` roots.

## Canonical Layer Status

Current canonical root:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/`

Key evidence:

- `summary.json`
- `canonical_calls_master.db`
- `canonical_preview.csv`
- `coverage_by_month.tsv`
- `db_scan_summary.tsv`
- `selected_by_db.tsv`
- `duplicate_conflicts.csv`

Validation status:

- Source audio: `64 867`
- Excluded manager-manager/no-ASR: `35`
- Actionable source audio: `64 832`
- ASR done actionable: `64 832`
- Full R+A actionable: `64 832`
- Missing ASR actionable: `0`
- Missing full R+A actionable: `0`
- Validation passed: `true`

Cleanup implication:

- This layer can be used as replacement evidence for older processing folders.
- It does not by itself authorize deletion of old folders; old artifacts still need archive/delete manifest rows with replacement artifact references.

## Active Artifacts To Keep Current

Treat these roots as current and protected:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/`
- `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`
- `stable_runtime/transcript_quality_stage14_comparison_20260510_v8_frozen_gate/`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/`
- `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/`
- `stable_runtime/crm_writeback_quality_gate_20260510_v12_stage40_new_single_contacts/`
- `stable_runtime/crm_writeback_quality_gate_20260510_v13_stage20_remaining_after_readback/`
- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T141007Z/`
- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T175140Z/`
- `stable_runtime/amocrm_runtime/contact_writebacks/20260510T180418Z/`
- `stable_runtime/amo_writeback_queue_20260510_v2_production/`

Treat these docs as current operational references:

- `docs/CURRENT_STATUS_2026-05-10.md`
- `docs/AMO_WRITEBACK_STAGE40_READBACK_REPORT_2026-05-10.md`
- `docs/AMO_WRITEBACK_PRODUCTION_LOOP_2026-05-10.md`
- `docs/CRM_TEXT_QUALITY_STAGE20_PLAN_2026-05-10.md`
- `docs/CRM_WRITEBACK_DEFECT_CLASS_MAP_2026-05-10.md`
- `docs/THREAT_MODEL.md`

## Superseded But Keep As Evidence

These are not current source-of-truth layers, but should not be deleted by Stage2 draft policy:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v2_crm_text_quality/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v3_crm_text_quality/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v4_crm_text_quality_strict/`
- Earlier `stable_runtime/crm_writeback_quality_gate_20260510_v1` through `v9` roots, unless superseded and not cited by any audit.
- Earlier `stable_runtime/transcript_quality_stage15_export_gate_20260510_v1` through `v10` roots.
- `stable_runtime/canonical_master_20260509_*` roots.
- Old April `sales_master_export_*` roots.

Stage2 action for these: index as superseded/evidence first. Move/archive only with a manifest that names the replacement current artifact and records whether an audit/result still cites the old path.

## Quarantine Buckets

### AMO Stage1 Production Queue

Current queue root:

- `stable_runtime/amo_writeback_queue_20260510_v2_production/`

Bucket state:

| Bucket | Rows | Stage2 decision |
|---|---:|---|
| `ready_single_contact_not_written` | `0` | no live-write candidate remains |
| `already_written` | `53` | keep as completed/evidence |
| `needs_manager_review_multi_contact` | `12` | quarantine; manager selects correct AMO contact |
| `blocked_contact_id_mismatch` | `1` | quarantine; resolve AMO/source linkage drift |
| `needs_text_quality_review` | `3` | quarantine; manual CRM text review or fresh strict rebuild |
| `deferred_non_sales_or_service` | `0` | empty |

Stage2 rule: only `ready_single_contact_not_written` may feed a future live stage. All other non-empty buckets are quarantine/manual-review buckets.

### Strict Export Manual Review

Manual review root:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/review_queues/manual_review_contacts_current.csv`

Rows: `12 297`.

Main groups to quarantine/manual review:

- contentful calls with manual review required;
- no AMO contact ID / orphan policy unresolved;
- service/existing-client/technical contexts that are not new sales leads;
- passive or closing next steps;
- multiple AMO contact IDs;
- CRM text quality blockers such as closure downgrade or ellipsis/truncation.

Stage2 rule: these rows should not be reintroduced through ad-hoc CSV edits. They need an explicit resolution path, a fresh strict export, and the standard gates.

### Bot/Autonomous Quarantine

Current Stage15 readiness explicitly keeps autonomous bot production blocked:

- `bot_autonomous_production_ready=false`
- blocker: `over_sanitization_queue_requires_rop_review_before_autonomous_bot`

Stage2 rule: CRM-internal manager-assist text and bot-safe output remain separate layers. Do not reuse CRM writeback text as bot/KB allowlist content.

## AMO Writeback Stage1 Closure

Stage1 live evidence:

- Stage20 earlier run wrote `20` rows: `20260510T141007Z`.
- Stage40 live A wrote `20` rows: `20260510T175140Z`.
- Stage40 live B wrote `20` rows: `20260510T180418Z`.
- Readback gates for Stage40 A and B passed, including expected-count readbacks.
- Combined report records `60` unique AMO contact phones across the three staged runs.

Stage1 closure evidence:

- Queue rebuilt from current strict `69`-row AMO-ready layer.
- Safe not-written bucket is empty: `ready_single_contact_not_written=0`.
- Remaining non-completed rows are quarantine/review classes.

Conclusion: Stage1 is complete and should stop here. Larger Stage50/full writeback is out of scope and not authorized by Stage2 cleanup docs.

## Cleanup / Archive Rules

Stage2 may prepare classification docs or dry-run manifests, but must not delete or mutate active artifacts.

Before any archive/move/delete step:

- Verify `stable_runtime/CANONICAL_EXPORT.txt` still points to the intended current layer.
- Verify current summaries are readable and validation still passes.
- Verify every candidate has a replacement artifact, preferably the current canonical DB or strict export.
- Verify no audit pack/result still depends on the exact path as its only evidence.
- Produce a dry-run manifest with `path`, `size`, `reason`, `replacement_artifact`, `safe_to_archive`, and `requires_manual_approval`.
- Get explicit owner approval for physical move/delete.

Never auto-clean these without separate approval:

- raw audio root `2026-03-09--26`;
- active canonical/current strict export roots listed above;
- AMO runtime/writeback/readback evidence;
- Tallanto/AMO credentials, `.env*`, tokens, secrets;
- audit evidence cited by current reports.

## Gate Checklist Before Next Pipeline Stage

Before building a new candidate stage:

- Canonical DB validation passed and matches expected counts.
- Phone-chain layer is from `insight_readiness_report_after_quality_backfill_20260510_v1` or a newer explicitly approved rebuild.
- Stage15 passed and `crm_quality_writeback_ready=true`.
- Current export pointer is not regressed to old April or early v1-v4 layers.
- CRM writeback quality gate has `blocking_rows=0` for the exact input CSV.
- Queue builder puts manual-review/text-quality rows before `already_written`.
- Real-tunnel dry-run passes for the staged scope.
- Live writeback uses explicit confirmation and expected count guards.
- Readback evaluates the expected number of rows and compares target field values.

## Existing Verification Evidence

No new tests were run for this docs-only update. Existing evidence from current reports:

- Canonical build report: full suite was reported as `611 passed, 1 warning` when the canonical master tooling was introduced.
- Stage1 audit pack: AMO queue/readback/writeback guard tests reported `36 passed`.
- Stage40 readback report: staged live A/B readback gates passed.
- CRM quality gate v10: `69` allow, `0` blockers.
- Stage15 v11: `passed=true`.

## Open Items

- Manual/manager resolution for `12` multi-contact rows.
- Linkage investigation for `1` contact-id mismatch row.
- CRM text review for `3` review-marker rows.
- Manual-review strategy for the broader `12 297` row queue.
- Separate archive/delete dry-run manifest if cleanup proceeds.
- Bot autonomous over-sanitization review before any autonomous bot production claim.
