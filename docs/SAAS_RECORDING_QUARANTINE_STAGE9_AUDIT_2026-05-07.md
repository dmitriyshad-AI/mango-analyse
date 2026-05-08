# SaaS Stage 9: Recording Quarantine Package Audit

Дата: 2026-05-07

## Цель этапа

Stage 9 закрывает цепочку Stage 1-9 для SaaS/productization ветки: из 5 проверенных `would_import` записей Stage 8 собран materialized quarantine package в product appliance.

Граница этапа:

- можно писать только под `product_appliance/recording_quarantine_stage9`;
- нельзя копировать аудио в legacy/current processing folders;
- нельзя писать runtime DB/product DB;
- нельзя трогать `stable_runtime`;
- нельзя запускать ASR/R+A;
- нельзя писать AMO/Tallanto.

## Проверка входа

Входной bridge plan:

- `recording_bridge_plan_stage8_legacy_source_check.json`
- `would_import`: 5
- `blocked`: 0
- `checksum_verified`: 5
- `source_audio_indexed`: 63,779
- `already_present_audio`: 0
- `validation_ok`: true

Вывод: Stage 8 пригоден как вход для Stage 9.

## Что добавлено

- `src/mango_mvp/productization/recording_quarantine_package.py`
  - normalizes Stage 8 wrapper report into plain bridge plan;
  - builds quarantine import plan using existing quarantine code;
  - validates all bridge source paths under product root;
  - validates all plan/materialized paths under product root;
  - blocks any `stable_runtime` reference;
  - materializes package with copy/hardlink modes via guarded wrapper.
- `scripts/mango_office_recording_quarantine_package.py`
  - `plan`: guarded bridge -> quarantine plan + metadata;
  - `materialize`: guarded plan -> materialized package audit.
- `tests/test_productization_recording_quarantine_package.py`
  - nested Stage 8 bridge report support;
  - plan + materialize + idempotency;
  - CLI plan/materialize;
  - outside product-root guards.

## Safety Contract

Stage 9 reports:

- `materialize_audio_into_quarantine`: true only for package materialization;
- `copy_audio_to_legacy_source`: false;
- `product_db_writes`: false;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false.

## Plan Command

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_quarantine_package.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  plan \
  --bridge-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_bridge_stage8/recording_bridge_plan_stage8_legacy_source_check.json \
  --package-root _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9 \
  --quarantine-dir _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/audio \
  --metadata-csv _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/metadata.csv \
  --normalized-bridge-plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/normalized_bridge_plan_stage9.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/recording_quarantine_plan_stage9.json
```

Plan result:

- `total_bridge_items`: 5
- `ready`: 5
- `blocked`: 0
- `metadata_rows`: 5
- `ready_total_mb`: 2.14
- `target_filename_collisions`: 0
- `path_audit.bridge_plan.blocked`: 0
- `path_audit.quarantine_plan.blocked`: 0
- `validation_ok`: true

Metadata:

- `metadata.csv`: 6 lines including header
- data rows: 5
- fields include source/target audio paths, phone, manager, started_at, call_id, recording_id, duration, checksum, tenant/provider.

## Materialization

Initial materialization copied the 5 ready files into the quarantine package:

- `copied`: 5
- `blocked`: 0
- `target_audio_files`: 5
- `checksum_mismatch_files`: 0
- `missing_expected_files`: 0
- `unreferenced_audio_files`: 0

The final persisted materialization audits were regenerated after a summary-field fix, so they correctly show the current idempotent state:

- `already_present`: 5
- `copied`: 0
- `blocked`: 0
- `expected_ready_files`: 5
- `target_audio_files`: 5
- `target_total_mb`: 2.14
- `checksum_verified_files`: 5
- `checksum_mismatch_files`: 0
- `missing_expected_files`: 0
- `zero_size_files`: 0
- `unreferenced_audio_files`: 0
- `validation_ok`: true
- `path_audit.materialized.blocked`: 0

Materialization command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_quarantine_package.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  materialize \
  --plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/recording_quarantine_plan_stage9.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/recording_quarantine_materialization_stage9_audit.json \
  --mode copy
```

Idempotency command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_recording_quarantine_package.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  materialize \
  --plan _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/recording_quarantine_plan_stage9.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9/recording_quarantine_materialization_stage9_idempotency_audit.json \
  --mode copy
```

## Package Contents

Package root:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/recording_quarantine_stage9
```

Files:

- `metadata.csv`
- `normalized_bridge_plan_stage9.json`
- `recording_quarantine_plan_stage9.json`
- `recording_quarantine_plan_stage9__materialize_input.json`
- `recording_quarantine_materialization_stage9_audit.json`
- `recording_quarantine_materialization_stage9_idempotency_audit.json`
- `audio/*.mp3`: 5 files

Audio files:

```text
45,792   2026-05-07__17-02-24__79214682746__mango_23_MjY2OTY5MTg4ODQ=.mp3
693,504  2026-05-07__17-08-59__79161907164__mango_26_MjY2OTY5OTEzMTY=.mp3
1,234,656 2026-05-07__17-12-16__79161907164__mango_26_MjY2OTcwMjcwNjk=.mp3
41,760   2026-05-07__17-14-31__79649593738__mango_23_MjY2OTcwNTExOTc=.mp3
231,840  2026-05-07__17-19-21__79052157579__mango_26_MjY2OTcxMDI2MTY=.mp3
```

Manager distribution:

- manager `23`: 2
- manager `26`: 3

## Product DB Integrity

After Stage 9:

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `job_runs`: 5
- `schema_migrations`: 4
- known warnings: 3 pending owner mappings

Stage 9 did not change product DB or runtime DB.

## Tests

Focused Stage 9/productization package gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_recording_quarantine_package.py \
  tests/test_productization_quarantine_import.py \
  tests/test_productization_quarantine_import_script.py \
  tests/test_productization_quarantine_materialize_script.py \
  tests/test_productization_recording_download_bridge.py \
  tests/test_productization_recording_capture_download.py
```

Result: `24 passed, 1 warning`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `142 passed, 1 warning`.

Insight gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_insight_readiness.py tests/test_pilot_extraction.py tests/test_outcome_linker.py tests/test_llm_review.py
```

Result: `32 passed`.

## Вывод

Stage 9 закрыт. У нас есть полноценный inspectable product-appliance quarantine package на 5 Mango записей:

1. normalized bridge input;
2. guarded quarantine import plan;
3. metadata.csv;
4. materialized audio package;
5. checksum/idempotency audit;
6. product DB integrity proof.

Следующий шаг уже вне этапов 1-9: isolated test ingest в disposable/test SQLite DB или product appliance DB extension, без runtime DB и без ASR/R+A, чтобы проверить end-to-end import semantics перед любым production ingest.
