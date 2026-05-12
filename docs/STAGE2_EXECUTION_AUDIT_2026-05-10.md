# Stage2 Execution Audit, 2026-05-10

## Scope

Выполнен второй этап наведения порядка и промышленного контура обработки звонков без физического удаления данных и без новых live CRM writes.

## Cleanup

Создан quarantine root:

- `_cleanup_quarantine_20260510_stage2/`

Содержимое:

- `MANIFEST.csv` - 30 перемещенных safe/superseded items.
- `README.md` - политика карантина и список protected anchors.
- `restore_one.sh` - восстановление одного original path без перезаписи существующего пути.
- `SUMMARY.tsv` - 30 items, logical size about 1.15 GB.

Принцип: данные не удалены, только вынесены из активной структуры. Физическое удаление требует отдельного решения владельца.

Protected/current anchors не перемещались:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/`
- `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/`
- `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/`
- `stable_runtime/amo_writeback_queue_20260510_v2_production/`
- AMO live/readback evidence `20260510T141007Z`, `20260510T175140Z`, `20260510T180418Z`.

## Current Pointer

`stable_runtime/CANONICAL_EXPORT.txt` now points to:

```text
sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict
```

## Industrial Call Processing Readiness

Добавлен read-only gate:

- Code: `src/mango_mvp/productization/call_processing_readiness.py`
- CLI: `scripts/mango_office_call_processing_readiness.py`
- Tests: `tests/test_productization_call_processing_readiness.py`
- Report: `stable_runtime/call_processing_readiness_20260510_stage2/report.json`

Боевой результат:

- Gates: `16/16` passed.
- `canonical_actionable_calls=64832`.
- `canonical_missing_asr=0`.
- `canonical_missing_ra=0`.
- `amo_ready_rows=69`.
- `safe_writeback_pending_rows=0`.
- `stage1_writeback_complete=true`.
- `processing_pipeline_ready=true`.

Gate связывает в один контур:

- current export pointer;
- canonical DB validation;
- ASR/R+A coverage;
- Stage15 CRM readiness;
- CRM writeback quality gate;
- AMO writeback queue;
- expected-count readback evidence;
- cleanup quarantine manifest.

## Mango Capture / Stage4 Productization

Автономно выполнены безопасные rechecks:

1. Controlled capture ingest plan:
   - `_local_archive_mango_api_downloads_20260507/product_appliance/controlled_capture_ingest_stage4/controlled_capture_ingest_stage2_recheck_20260510.json`
   - Result: `validation_ok=true`, blocked `0`, no product DB writes, no audio downloads, no ASR/R+A, no CRM writes.

2. Processing lifecycle recheck:
   - `_local_archive_mango_api_downloads_20260507/product_appliance/processing_lifecycle_stage5/processing_lifecycle_stage2_recheck_20260510.json`
   - Result: `validation_ok=true`, 21 capture items already in handoff manifest, no pending ASR handoff.

3. Processing acceptance gates with current quality evidence:
   - `_local_archive_mango_api_downloads_20260507/product_appliance/processing_acceptance_gates/processing_acceptance_gates_stage2_recheck_20260510.json`
   - Result: `validation_ok=true`, blocked `0`, warnings `1` for CRM/Tallanto snapshot evidence.

Дополнительно усилен dry-run scheduler: archived Mango raw payload now replays through the same mapper/planner as live shadow poll without provider call. This catches shape/dedupe/no-recording issues without Mango credentials.

Updated files:

- `src/mango_mvp/productization/scheduler_runtime.py`
- `tests/test_productization_scheduler_runtime.py`
- `.env.example` now documents `MANGO_OFFICE_BASE_URL`, `MANGO_OFFICE_API_KEY`, `MANGO_OFFICE_API_SALT`.

## Project Inventory

Fresh inventory after quarantine:

- `stable_runtime/project_inventory_20260510_stage2_after_quarantine/summary.json`

Key counters:

- Project size: about 73.3 GB.
- DB files: 267.
- DB total size: about 24.3 GB.
- Archive candidate rows remaining: 88.
- Archive candidate size: about 16.0 GB.

Interpretation: Stage2 moved only the low-risk first slice. Larger archive candidates remain for a separate owner-approved archive/delete pass.

## AMO Writeback Status

No new live AMO writeback was performed in this Stage2 pass.

Stage1 remains closed:

- `ready_single_contact_not_written=0` in `amo_writeback_queue_20260510_v2_production`.
- Remaining rows are review/block buckets, not live candidates.

## Tests

Focused regression:

```text
69 passed, 1 warning
```

Full suite:

```text
797 passed, 1 warning
```

One stale test was corrected to enforce the current no-ellipsis CRM text policy: long AMO text fields must end with `[сжато]`, not `...`.

## Audit Pack

Prepared compact audit pack:

- `audits/_inbox/stage2_cleanup_pipeline_20260510_v1/`

It contains small copies of readiness, inventory, quarantine and productization summary artifacts. Large DB/audio files are intentionally excluded.

## Current Limitations

- Quarantine is not deletion. Owner approval is needed before removing it.
- Product acceptance has one warning: CRM/Tallanto snapshot evidence should be refreshed before external demo claims.
- Productization ASR execution remains fail-closed/dry-run unless explicitly approved.
- Manual-review queues remain open and should not feed AMO writeback without fresh gates.

## Recommended Next Actions

1. Ask owner to review quarantine manifest and decide whether to keep, restore, archive externally, or delete later.
2. Refresh AMO/Tallanto read-only snapshots for product acceptance warning.
3. Build the next product dashboard/readiness view on top of `call_processing_readiness_v1`.
4. If continuing CRM work, resolve the remaining 12 multi-contact, 1 contact-id mismatch and 3 text-quality review rows through a new staged queue.
5. Do not start Stage50/full AMO writeback from current strict layer: current safe bucket is empty.
