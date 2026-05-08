# SaaS Stage 12: Processing Handoff Contract Audit

Дата: 2026-05-07

## Цель этапа

Stage 12 добавляет dry-run processing handoff contract для будущего ASR-worker:

```text
isolated asset DB -> ready_for_asr JSONL manifest
```

Граница этапа:

- читаем Stage 11 `recording_asset_ingest_stage11.sqlite` только read-only;
- пишем только JSON/JSONL под `product_appliance/processing_handoff_stage12`;
- не запускаем ASR;
- не запускаем R+A;
- не пишем runtime DB;
- не трогаем `stable_runtime`;
- не пишем AMO/Tallanto/CRM.

## Что добавлено

- `src/mango_mvp/productization/processing_handoff.py`
  - читает `captured_recording_assets` из isolated asset DB;
  - выбирает `quarantined_ready` assets;
  - проверяет audio path под `product_appliance`;
  - блокирует `stable_runtime` references;
  - проверяет `.mp3`, file existence, zero-size, checksum;
  - строит deterministic `queue_item_id`;
  - рассчитывает planned ASR output paths;
  - пишет JSONL manifest только для `PLAN_ASR_HANDOFF`;
  - возвращает audit с `BLOCK_ASR_HANDOFF` / `SKIP_NOT_READY_FOR_ASR`.
- `scripts/mango_office_processing_handoff.py`
  - CLI для Stage 12 dry-run handoff.
- `tests/test_productization_processing_handoff.py`
  - ready assets -> manifest;
  - idempotent manifest hash;
  - missing audio block;
  - checksum mismatch block;
  - runtime/outside path guards;
  - CLI audit.

## Safety Contract

Stage 12 reports:

- `product_db_writes`: false;
- `asset_db_writes`: false;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `downloads_audio`: false;
- `copies_audio`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false;
- `write_tallanto`: false.

## Handoff Contract

Manifest item status:

```text
ready_for_asr
```

Required fields:

- `queue_item_id`
- `tenant_id`
- `provider`
- `event_key`
- `recording_id`
- `audio_path`
- `checksum_sha256`
- `planned_outputs`

Planned outputs:

- `transcript_json`
- `transcript_txt`
- `asr_audit_json`

Worker requirements:

- verify `audio_path_exists`;
- verify `checksum_sha256`;
- require explicit runtime target approval before any real ASR execution.

## Real Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_processing_handoff.py
```

Result:

- `source_assets_seen`: 21
- `selected_assets`: 21
- `PLAN_ASR_HANDOFF`: 21
- `ready_for_asr`: 21
- `manifest_rows`: 21
- `blocked`: 0
- `skipped_not_ready`: 0
- `warnings`: 0
- `validation_ok`: true
- `manifest_sha256`: `e74e09da419e0314e712bbde4a170e952b195e7ed473d7da9fb610b5a8f05c01`

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/processing_handoff_stage12/asr_handoff_manifest_stage12.jsonl
_local_archive_mango_api_downloads_20260507/product_appliance/processing_handoff_stage12/asr_handoff_stage12_audit.json
```

## Idempotency Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_processing_handoff.py --idempotency-out
```

Result:

- `PLAN_ASR_HANDOFF`: 21
- `ready_for_asr`: 21
- `manifest_rows`: 21
- `manifest_sha256`: `e74e09da419e0314e712bbde4a170e952b195e7ed473d7da9fb610b5a8f05c01`
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true

The manifest hash stayed stable.

## Product DB Integrity

Main product appliance DB was checked after Stage 12. Stage 12 did not write to it.

Result:

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `job_runs`: 5
- `schema_migrations`: 4
- known warnings: 3 pending owner mappings

## Tests

Focused handoff gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_processing_handoff.py \
  tests/test_productization_recording_asset_ingest.py
```

Result: `11 passed`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `153 passed, 1 warning`.

The warning is the existing external urllib3/LibreSSL warning.

## Вывод

Stage 12 закрыт. У нас появился безопасный processing handoff layer: 21 Mango recordings уже могут быть представлены как deterministic `ready_for_asr` manifest для будущего ASR worker, но сам ASR/R+A по-прежнему не запускается и runtime DB не меняется.

Следующий безопасный этап: добавить dry-run worker pack builder, который из `asr_handoff_manifest_stage12.jsonl` собирает переносимый ASR input pack под product appliance, без выполнения ASR.
