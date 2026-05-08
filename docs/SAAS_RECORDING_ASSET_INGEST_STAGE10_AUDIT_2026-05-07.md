# SaaS Stage 10: Isolated Recording Asset Ingest Audit

Дата: 2026-05-07

## Цель этапа

Stage 10 проверяет end-to-end import semantics для Stage 9 Mango quarantine package:

```text
metadata.csv + audio/*.mp3 -> isolated recording asset SQLite -> idempotency/audit
```

Граница этапа:

- можно писать только под `product_appliance/recording_quarantine_stage10`;
- нельзя писать runtime DB;
- нельзя трогать `stable_runtime`;
- нельзя запускать ASR/R+A;
- нельзя писать AMO/Tallanto/CRM;
- нельзя менять текущие batch/start/run-ui scripts.

## Что добавлено

- `src/mango_mvp/productization/recording_asset_ingest.py`
  - читает Stage 9 `metadata.csv`;
  - валидирует required keys: `tenant_id`, `provider`, `event_key`, `provider_call_id`, `recording_id`, `filename`;
  - проверяет, что audio paths остаются внутри `product_appliance`;
  - блокирует любые `stable_runtime` references;
  - проверяет `.mp3`, file existence, size, checksum;
  - создает isolated productization SQLite schema:
    - `recording_asset_schema_migrations`;
    - `recording_import_packages`;
    - `captured_recording_assets`;
    - `recording_asset_ingest_runs`;
  - делает идемпотентный upsert:
    - `INGEST_RECORDING_ASSET`;
    - `SKIP_ALREADY_INGESTED`;
    - `UPDATE_RECORDING_ASSET`;
    - `BLOCK_RECORDING_ASSET_INGEST`.
- `scripts/mango_office_recording_asset_ingest.py`
  - CLI для Stage 10 ingest/audit;
  - default paths указывают на Stage 9 package и Stage 10 isolated DB.
- `tests/test_productization_recording_asset_ingest.py`
  - import + idempotency;
  - checksum mismatch block;
  - duplicate metadata block;
  - runtime/outside path guards;
  - CLI audit write.

## Safety Contract

Stage 10 reports:

- `isolated_productization_db_writes`: true;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `downloads_audio`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false;
- `write_tallanto`: false;
- `legacy_call_records_used`: false;
- `product_calls_updated`: false.

## Real Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_asset_ingest.py --replace-db
```

Result:

- `validation_ok`: true
- `metadata_rows`: 5
- `planned_assets`: 5
- `INGEST_RECORDING_ASSET`: 5
- `blocked`: 0
- `warnings`: 0
- `db_assets_for_package`: 5
- `status_counts`: `quarantined_ready: 5`
- `manager_counts`: `23: 2`, `26: 3`
- checksum mismatches: 0
- missing audio: 0
- paths outside product root: 0
- paths outside audio dir: 0

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage10/recording_asset_ingest_stage10_audit.json
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage10/recording_asset_ingest_stage10.sqlite
```

## Idempotency Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_asset_ingest.py \
  --allow-existing-db \
  --idempotency-out
```

Result:

- `validation_ok`: true
- `SKIP_ALREADY_INGESTED`: 5
- `inserted`: 0
- `updated`: 0
- `blocked`: 0
- `warnings`: 0
- `db_assets_for_package`: 5
- `ingest_runs`: 2

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage10/recording_asset_ingest_stage10_idempotency_audit.json
```

## Product DB Integrity Check

Main product appliance DB was not used as the Stage 10 ingest target.

Integrity command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage10/product_db_integrity_stage10_audit.json
```

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
  tests/test_productization_recording_quarantine_package.py
```

Result: `9 passed`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `147 passed, 1 warning`.

The warning is the existing external urllib3/LibreSSL warning and is not caused by Stage 10.

## Вывод

Stage 10 закрыт. У нас появился isolated asset-ingest слой: 5 Mango recordings из Stage 9 теперь представлены не только как файлы quarantine package, но и как проверяемые productization records с checksum/path/idempotency audit.

Следующий безопасный этап: масштабировать ту же цепочку с 5 до всех 21 ready capture inbox items, затем повторить quarantine package + isolated asset ingest на расширенном наборе.
