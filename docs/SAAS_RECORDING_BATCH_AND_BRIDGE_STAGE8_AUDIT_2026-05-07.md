# SaaS Stage 8: Recording Batch And Bridge Dry-Run Audit

Дата: 2026-05-07

## Цель этапа

Stage 8 расширяет Stage 7 с одного скачанного Mango recording до малой контролируемой пачки и строит dry-run bridge/import plan.

Граница этапа:

- разрешено скачать малую пачку в product appliance;
- разрешено построить JSON/CSV bridge plan;
- запрещено писать в runtime DB;
- запрещено копировать аудио в legacy/current processing folders;
- запрещено запускать ASR/R+A;
- запрещено писать в AMO/Tallanto.

## Проверка входа

Stage 7 verify audit:

- `manifest_rows`: 1
- `downloaded_latest_events`: 1
- `recordings_dir_mp3_files`: 1
- `checksum_mismatches`: 0
- `missing_files`: 0
- `unreferenced_audio_files`: 0
- `validation_ok`: true

Вывод: Stage 7 пригоден как вход для Stage 8.

## Что добавлено и исправлено

Добавлено:

- `src/mango_mvp/productization/recording_download_bridge.py`
  - конвертирует Stage 7/8 download manifest в `capture_manifest_v1` view;
  - строит read-only bridge plan через существующий `pipeline_bridge`;
  - пишет JSON/CSV plan под product root;
  - не копирует аудио, не пишет DB, не запускает ASR/R+A.
- `scripts/mango_office_recording_bridge_dry_run.py`
  - CLI для download manifest -> bridge plan.
- `tests/test_productization_recording_download_bridge.py`
  - conversion of `DOWNLOADED_RECORDING` and `SKIP_ALREADY_DOWNLOADED`;
  - CLI report;
  - output path guard.

Исправлено:

- `recording_capture_download` audit теперь считает `SKIP_ALREADY_DOWNLOADED` как доступный локальный audio asset, если файл существует и проходит checksum/size validation.
- Повторный запуск `--limit N` больше не теряет уже скачанную запись из available-состояния.

## Stage 8 Preflight

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  run \
  --source-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage8_preflight.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage8_preflight_audit.json \
  --limit 5 \
  --sleep-sec 0
```

Результат:

- `execute`: false
- `selected_items`: 5
- `skip_already_downloaded`: 1
- `PLAN_RECORDING_DOWNLOAD`: 4
- новых файлов: 0
- `validation_ok`: true

## Stage 8 Controlled Execute

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  run \
  --source-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage8.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage8_audit.json \
  --execute \
  --limit 5 \
  --sleep-sec 0 \
  --timeout-sec 60 \
  --link-retries 3 \
  --rate-limit-sleep-sec 10
```

Результат:

- `selected_items`: 5
- `skip_already_downloaded`: 1
- `downloaded_recording`: 4
- `failed_download`: 0
- `downloaded_bytes_total`: 2,247,552
- `validation_ok`: true

Independent download audit:

- `available_latest_events`: 5
- `downloaded_latest_events`: 4
- `recordings_dir_mp3_files`: 5
- `recordings_dir_total_bytes`: 2,247,552
- `checksum_mismatches`: 0
- `missing_files`: 0
- `zero_size_files`: 0
- `local_paths_outside_root`: 0
- `unreferenced_audio_files`: 0
- `validation_ok`: true

Manager distribution:

- manager `23`: 2
- manager `26`: 3

Total audio duration in bridge plan: 561.888 sec.

## Bridge Dry-Run

Product-only bridge command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_bridge_dry_run.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage8.jsonl \
  --capture-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/capture_manifest_from_downloads_stage8.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/recording_bridge_plan_stage8.json \
  --csv-out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/recording_bridge_plan_stage8.csv \
  --source-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/empty_legacy_source_index \
  --tolerance-sec 120
```

Результат:

- `download_manifest_rows`: 5
- `latest_available_events`: 5
- `converted_capture_manifest_rows`: 5
- `bridge_total_manifest_events`: 5
- `would_import`: 5
- `blocked`: 0
- `validation_ok`: true

Read-only legacy source check:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_bridge_dry_run.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage8.jsonl \
  --capture-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/capture_manifest_from_downloads_stage8_legacy_source_check.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/recording_bridge_plan_stage8_legacy_source_check.json \
  --csv-out _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/recording_bridge_plan_stage8_legacy_source_check.csv \
  --source-dir 2026-03-09--26 \
  --tolerance-sec 120
```

Результат:

- `source_audio_indexed`: 63,779
- `db_calls_indexed`: 0
- `checksum_verified`: 5
- `would_import`: 5
- `already_present_audio`: 0
- `blocked`: 0
- `validation_ok`: true

DB не читалась в этом check; `db_paths` пустой.

## Product DB Integrity

После Stage 8:

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `job_runs`: 5
- `schema_migrations`: 4
- known warnings: 3 pending owner mappings

Stage 8 не менял product DB и не касался runtime DB.

## Тесты

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_recording_capture_download.py \
  tests/test_productization_recording_download_bridge.py \
  tests/test_productization_mango_recordings.py \
  tests/test_productization_recording_capture_plan.py
```

Результат: `14 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Результат: `138 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_insight_readiness.py tests/test_pilot_extraction.py tests/test_outcome_linker.py tests/test_llm_review.py
```

Результат: `32 passed`.

## Вывод

Stage 8 закрыт. Product appliance теперь умеет:

1. безопасно расширить controlled recording download до малой пачки;
2. валидировать 5 локальных mp3 через size/checksum/audio metadata;
3. конвертировать download manifest в bridge-ready `capture_manifest_v1`;
4. строить JSON/CSV import plan;
5. проверять legacy audio folder read-only без DB/runtime writes.

Следующий безопасный Stage 9: quarantine import package dry-run/materialization для этих 5 `would_import` записей в отдельной product appliance папке, без запуска ASR/R+A и без записи в runtime DB.
