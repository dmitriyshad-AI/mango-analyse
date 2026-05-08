# SaaS Stage 7: Controlled Recording Download Audit

Дата: 2026-05-07

## Цель этапа

Stage 7 переводит Stage 6 manifest из состояния `PLAN_DOWNLOAD_DRY_RUN` в контролируемый download-контур.
Это первый SaaS-level шаг, где разрешено скачать один аудиофайл из Mango, но только в изолированную product appliance папку.

## Проверка входа

Входной Stage 6 manifest:

- `recording_capture_plan_stage6.jsonl`
- `items`: 21
- `PLAN_DOWNLOAD_DRY_RUN`: 21
- `SKIP_DUPLICATE_RECORDING`: 0
- `BLOCK_MISSING_RECORDING_REF`: 0
- `target_paths_outside_root`: 0
- `validation_ok`: true

Вывод: вход пригоден для controlled download.

## Что добавлено

- `src/mango_mvp/productization/recording_capture_download.py`
  - читает только Stage 6 JSONL manifest;
  - пишет отдельный download manifest;
  - поддерживает `--execute` opt-in;
  - поддерживает `--limit` и `--manager-ref`;
  - валидирует output paths под product root;
  - считает checksum/size и, если доступно, audio metadata;
  - audit проверяет missing/zero-size/checksum mismatch/outside root/unreferenced files.
- `scripts/mango_office_recording_capture_download.py`
  - `run`: preflight или controlled download;
  - `audit`: независимая проверка download manifest.
- `tests/test_productization_recording_capture_download.py`
  - fake downloader;
  - limited execute;
  - idempotency;
  - CLI preflight;
  - checksum mismatch audit;
  - output path guard.
- `tests/test_productization_mango_recordings.py`
  - fake HTTP проверка Mango recording signature;
  - fake streaming download.

## Граница безопасности

Контракт Stage 7:

- `download_audio`: true только при `--execute`;
- `product_db_writes`: false;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false.

Stage 7 не пишет в runtime DB, не трогает `stable_runtime`, не запускает ASR/R+A, не пишет в AMO/Tallanto.

## Preflight

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  run \
  --source-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage7_preflight.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage7_preflight_audit.json \
  --limit 1 \
  --sleep-sec 0
```

Результат:

- `execute`: false
- `selected_items`: 1
- `PLAN_RECORDING_DOWNLOAD`: 1
- скачано файлов: 0
- `validation_ok`: true

## Controlled Execute

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  run \
  --source-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage7.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage7_audit.json \
  --execute \
  --limit 1 \
  --sleep-sec 0 \
  --timeout-sec 60 \
  --link-retries 3 \
  --rate-limit-sleep-sec 10
```

Результат:

- `execute`: true
- `selected_items`: 1
- `DOWNLOADED_RECORDING`: 1
- `FAILED_DOWNLOAD`: 0
- `downloaded_bytes_total`: 45,792
- `validation_ok`: true
- `warnings`: 0

Скачанный файл:

- `recording_capture_downloads/recordings/20260507T140224Z__mgr_23__call_f7e7637c0dea__rec_4af0eb407e65.mp3`
- size: 45,792 bytes
- checksum: `0becf81e38ba02b3d6d1beba7a3fff8dc7535f24abc961c81917439315507360`
- codec: `mp3`
- duration: 11.448 sec
- channels: 2
- sample rate: 8000

## Independent Audit

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  audit \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recordings \
  --download-manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_manifest_stage7.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_downloads/recording_download_stage7_verify_audit.json
```

Результат:

- `manifest_rows`: 1
- `latest_unique_events`: 1
- `downloaded_latest_events`: 1
- `failed_latest_events`: 0
- `missing_files`: 0
- `zero_size_files`: 0
- `checksum_mismatches`: 0
- `local_paths_outside_root`: 0
- `unreferenced_audio_files`: 0
- `recordings_dir_mp3_files`: 1
- `recordings_dir_total_bytes`: 45,792
- `validation_ok`: true

## Product DB Integrity

После Stage 7:

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `job_runs`: 5
- `schema_migrations`: 4
- known warnings: 3 pending owner mappings

Stage 7 не изменял product DB и не касался runtime DB.

## Тесты

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_recording_capture_download.py \
  tests/test_productization_mango_recordings.py \
  tests/test_productization_recording_capture_plan.py
```

Результат: `11 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Результат: `135 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_insight_readiness.py tests/test_pilot_extraction.py tests/test_outcome_linker.py tests/test_llm_review.py
```

Результат: `32 passed`.

## Вывод

Stage 7 закрыт. Product appliance теперь умеет безопасно скачать запись из Mango по заранее проверенному manifest-у и доказать результат через checksum/size/audio audit.

Следующий безопасный этап: расширить controlled download с `--limit 1` до малой пачки, например `--limit 5`, затем построить download-to-bridge dry-run для подготовки к isolated ingest без запуска ASR/R+A.
