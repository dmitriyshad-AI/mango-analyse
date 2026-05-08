# Mango quarantine import dry-run audit

Дата: 2026-05-07

Цель: собрать безопасный import package plan из `pipeline_bridge_plan.json` без копирования аудио в рабочую папку, без записи в DB и без запуска ASR/R+A.

## Что добавлено

Новые файлы:

- `src/mango_mvp/productization/quarantine_import.py`
- `scripts/mango_office_quarantine_import_plan.py`
- `tests/test_productization_quarantine_import.py`
- `tests/test_productization_quarantine_import_script.py`

## Safety boundaries

Команда пишет только planning artifacts:

- `quarantine_import_plan.json`
- `metadata.csv`

Команда не делает:

- копирование файлов в `2026-03-09--26`;
- hardlink/symlink файлов;
- запись в `mango_mvp.db`;
- запись в `stable_runtime`;
- ASR/R+A;
- AMO/Tallanto writeback.

## Command

```zsh
PYTHONPATH=src python3 scripts/mango_office_quarantine_import_plan.py \
  --bridge-plan _local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.json \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/quarantine_import_plan.json
```

## Outputs

Plan:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/quarantine_import_plan.json
```

Metadata:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv
```

Planned quarantine audio dir:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/audio
```

Because this is dry-run, the audio dir was not created and no mp3 files were copied there.

## Result

```text
copy_mode = dry_run
total_bridge_items = 297
ready = 297
blocked = 0
skipped_non_import_status = 0
unique_target_filenames = 297
target_filename_collisions = 0
metadata_rows = 297
quarantine_audio_files = 0
ready_total_mb = 180.74
```

By day:

```text
2026-05-06 = 176
2026-05-07 = 121
```

Interpretation:

- Every bridge `would_import` row became a ready quarantine import row.
- Every source mp3 exists and checksum passed.
- All target filenames are deterministic and unique.
- `metadata.csv` is ready for a future materialized quarantine package.

## Metadata CSV fields

The CSV includes both legacy ingest-compatible fields and product/audit fields:

- `filename`
- `source_audio_path`
- `target_audio_path`
- `phone`
- `client_phone`
- `manager`
- `manager_name`
- `started_at`
- `start_time`
- `direction`
- `call_id`
- `record_id`
- `event_key`
- `provider_call_id`
- `recording_id`
- `duration_sec`
- `checksum_sha256`
- `source_size_bytes`
- `source`
- `tenant_id`
- `provider`

## Test gate

Command:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
53 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for local dry-run.

## Next recommended step

Next safe step: `materialized quarantine package`, but only after explicit approval.

That step should:

1. Create the quarantine audio directory.
2. Copy or hardlink the 297 source mp3 files using the deterministic target filenames.
3. Re-run metadata/audio validation against the materialized package.
4. Still not write to `mango_mvp.db`.
5. Still not launch ASR/R+A.

Only after the materialized package is manually inspected should a later task define a gated ingest into a separate test SQLite DB.
