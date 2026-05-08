# SaaS Stage 6: Recording Capture Dry-Run Audit

Дата: 2026-05-07

## Цель этапа

Stage 6 добавляет безопасный планировщик будущего скачивания записей Mango из `capture_inbox_items`.
Этап намеренно не скачивает аудио и не меняет текущий processing pipeline.

## Проверка предыдущего этапа

Перед Stage 6 проверен Stage 5 capture inbox/product DB контур:

- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `schema_migrations`: 4
- `job_runs`: 5
- `validation_ok`: true
- известные warnings: 3 pending owner mappings

Вывод: Stage 5 пригоден как вход для Stage 6.

## Что добавлено

- `src/mango_mvp/productization/recording_capture_plan.py`
  - read-only чтение `capture_inbox_items`;
  - построение JSONL manifest-а будущего захвата записей;
  - action-коды:
    - `PLAN_DOWNLOAD_DRY_RUN`;
    - `SKIP_DUPLICATE_RECORDING`;
    - `SKIP_EXISTING_FILE`;
    - `BLOCK_MISSING_RECORDING_REF`;
  - deterministic target path для будущего аудиофайла;
  - отдельный audit manifest-а.
- `scripts/mango_office_recording_capture_plan.py`
  - `build`: product DB -> dry-run manifest + audit JSON;
  - `audit`: проверка manifest-а без обращения к Mango.
- `tests/test_productization_recording_capture_plan.py`
  - dry-run safety flags;
  - duplicate/missing recording risk detection;
  - manager filter and limit;
  - CLI build/audit;
  - guard против output paths вне product root.

## Граница безопасности

Stage 6 соблюдает контракт:

- `product_db_writes`: false
- `runtime_db_writes`: false
- `stable_runtime_writes`: false
- `download_audio`: false
- `run_asr`: false
- `run_ra`: false
- `write_crm`: false

Дополнительно после аудита кода убран вызов миграций из planner-а: модуль теперь не пытается менять product DB, а только проверяет, что таблица `capture_inbox_items` уже существует.

## Реальный прогон

Команда build:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_plan.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  build \
  --recordings-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recordings \
  --manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6_audit.json
```

Результат:

- `inbox_items_seen`: 21
- `manifest_items`: 21
- `PLAN_DOWNLOAD_DRY_RUN`: 21
- `SKIP_DUPLICATE_RECORDING`: 0
- `SKIP_EXISTING_FILE`: 0
- `BLOCK_MISSING_RECORDING_REF`: 0
- `validation_ok`: true
- `warnings`: 0

Manager distribution:

- manager `26`: 13
- manager `23`: 6
- manager `202`: 2

Команда audit:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_capture_plan.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  audit \
  --manifest _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6.jsonl \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_capture_dry_run/recording_capture_plan_stage6_verify_audit.json
```

Результат:

- `items`: 21
- `duplicate_target_paths`: 0
- `target_paths_outside_root`: 0
- `existing_audio_files`: 0
- `blocked`: 0
- `validation_ok`: true

Проверка файлов:

- manifest lines: 21
- аудиофайлов в `recording_capture_dry_run/recordings`: 0
- в Stage 6 output появились только JSON/JSONL audit artifacts.

## Product DB integrity после Stage 6

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `calls_with_crm_owner`: 219
- `pending_owner_mappings`: 3
- `warnings`: 3

Stage 6 не изменил runtime DB/audio/transcripts и не запускал ASR/R+A.

## Тесты

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_recording_capture_plan.py tests/test_productization_capture_inbox.py
```

Результат: `8 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Результат: `128 passed, 1 warning`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_insight_readiness.py tests/test_pilot_extraction.py tests/test_outcome_linker.py tests/test_llm_review.py
```

Результат: `32 passed`.

## Вывод

Stage 6 закрыт. Мы получили проверяемый список из 21 записи, которые можно будет скачивать на следующем этапе, но пока скачивание не выполняется.

Следующий безопасный этап: controlled recording downloader в изолированную product appliance папку, сначала с `--limit 1`/`--dry-run false` только после ручного подтверждения, с checksum/size audit и без ASR/R+A/CRM/runtime DB writes.
