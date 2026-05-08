# SaaS Stage 11: Recording Capture Scale to 21 Audit

Дата: 2026-05-07

## Цель этапа

Stage 11 масштабирует безопасный Mango capture контур с первых 5 записей до всех `21` ready capture inbox items:

```text
capture plan 21 -> controlled download -> bridge dry-run -> quarantine package -> isolated asset ingest
```

Граница этапа:

- можно писать только под `product_appliance`;
- нельзя писать runtime DB;
- нельзя трогать `stable_runtime`;
- нельзя запускать ASR/R+A;
- нельзя писать AMO/Tallanto/CRM;
- legacy source `2026-03-09--26` использовался только read-only для bridge duplicate check.

## Input State

Stage 6 capture plan:

- `inbox_items_seen`: 21
- `manifest_items`: 21
- `PLAN_DOWNLOAD_DRY_RUN`: 21
- `blocked`: 0
- `validation_ok`: true

Source:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl
```

## Controlled Download

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py run \
  --source-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage11.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage11_audit.json \
  --execute \
  --limit 21 \
  --sleep-sec 0.5
```

Result:

- `selected_items`: 21
- `skip_already_downloaded`: 5
- `downloaded_recording`: 16
- `failed_download`: 0
- `blocked`: 0
- `downloaded_bytes_total`: 6,706,368
- `validation_ok`: true

Verify audit:

- `manifest_rows`: 21
- `latest_unique_events`: 21
- `available_latest_events`: 21
- `recordings_dir_mp3_files`: 21
- `missing_files`: 0
- `checksum_mismatches`: 0
- `zero_size_files`: 0
- `unreferenced_audio_files`: 0
- `local_paths_outside_root`: 0
- `validation_ok`: true

## Bridge Dry-Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_bridge_dry_run.py \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage11.jsonl \
  --capture-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage11/capture_manifest_from_downloads_stage11_legacy_source_check.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage11/recording_bridge_plan_stage11_legacy_source_check.json \
  --csv-out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage11/recording_bridge_plan_stage11_legacy_source_check.csv \
  --source-dir 2026-03-09--26
```

Result:

- `download_manifest_rows`: 21
- `converted_capture_manifest_rows`: 21
- `bridge_total_manifest_events`: 21
- `latest_available_events`: 21
- `would_import`: 21
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true

## Quarantine Package

Package root:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage11
```

Plan result:

- `total_bridge_items`: 21
- `ready`: 21
- `metadata_rows`: 21
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true

Materialization result:

- `copied`: 21
- `target_audio_files`: 21
- `target_total_mb`: 6.4
- `checksum_mismatch_files`: 0
- `missing_expected_files`: 0
- `unreferenced_audio_files`: 0
- `zero_size_files`: 0
- `validation_ok`: true

Idempotency materialization:

- `already_present`: 21
- `copied`: 0
- `blocked`: 0
- `validation_ok`: true

## Isolated Asset Ingest

Target DB:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage11/recording_asset_ingest_stage11.sqlite
```

Initial ingest:

- `metadata_rows`: 21
- `planned_assets`: 21
- `INGEST_RECORDING_ASSET`: 21
- `db_assets_for_package`: 21
- `status_counts`: `quarantined_ready: 21`
- `manager_counts`: `202: 2`, `23: 6`, `26: 13`
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true

Idempotency ingest:

- `SKIP_ALREADY_INGESTED`: 21
- `inserted`: 0
- `updated`: 0
- `blocked`: 0
- `warnings`: 0
- `ingest_runs`: 2
- `validation_ok`: true

## Product DB Integrity

Main product appliance DB was checked after Stage 11. Stage 11 did not use it as the asset ingest target.

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

Focused gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_recording_asset_ingest.py \
  tests/test_productization_recording_capture_download.py \
  tests/test_productization_recording_download_bridge.py \
  tests/test_productization_recording_quarantine_package.py
```

Result: `17 passed, 1 warning`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `147 passed, 1 warning`.

The warning is the existing external urllib3/LibreSSL warning.

## Вывод

Stage 11 закрыт. Все 21 ready Mango capture items теперь имеют:

1. controlled download manifest;
2. local mp3 files under product appliance;
3. read-only bridge plan against legacy source;
4. materialized quarantine package;
5. isolated `captured_recording_assets` rows;
6. checksum/path/idempotency audits.

Следующий безопасный этап: добавить processing handoff contract в productization-ветке: read-only/dry-run очередь `ready_for_asr` на основе isolated asset DB, без запуска ASR/R+A.
