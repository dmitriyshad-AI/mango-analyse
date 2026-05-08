# Mango pipeline bridge dry-run audit

Дата: 2026-05-07

Цель: проверить безопасный bridge между новым Mango capture staging и текущим legacy pipeline без записи в runtime-БД, без копирования аудио в рабочую папку и без запуска ASR/R+A.

## Что добавлено

Новые файлы:

- `src/mango_mvp/productization/pipeline_bridge.py`
- `scripts/mango_office_pipeline_bridge_dry_run.py`

Новые тесты:

- `tests/test_productization_pipeline_bridge.py`
- `tests/test_productization_pipeline_bridge_script.py`

Bridge делает только read-only dry-run:

1. Читает `capture_manifest.jsonl`.
2. Берет только latest entries.
3. Проверяет `status=downloaded`.
4. Проверяет наличие локального mp3.
5. Проверяет размер файла.
6. Проверяет `checksum_sha256`.
7. Проверяет наличие `duration_sec`.
8. Строит proposed legacy filename и metadata mapping.
9. Сверяет с рабочей аудио-папкой по `phone + started_at ± tolerance`.
10. Сверяет с SQLite DB через read-only URI `mode=ro`.
11. Пишет JSON/CSV import plan.

Bridge не делает:

- копирование файлов в `2026-03-09--26`;
- запись в `mango_mvp.db`;
- запись в `stable_runtime`;
- запуск ASR/R+A;
- AMO/Tallanto writeback.

## Команда

```zsh
PYTHONPATH=src python3 scripts/mango_office_pipeline_bridge_dry_run.py \
  --manifest _local_archive_mango_api_downloads_20260507/capture_manifest.jsonl \
  --source-dir 2026-03-09--26 \
  --db mango_mvp.db \
  --out _local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.json \
  --csv-out _local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.csv \
  --tolerance-sec 120
```

## Inputs

Manifest:

```text
_local_archive_mango_api_downloads_20260507/capture_manifest.jsonl
```

Source audio dir:

```text
2026-03-09--26
```

Read-only DB:

```text
mango_mvp.db
```

## Output

JSON:

```text
_local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.json
```

CSV:

```text
_local_archive_mango_api_downloads_20260507/pipeline_bridge_plan.csv
```

## Result

```text
total_manifest_events = 297
checksum_verified = 297
checksum_skipped = 0
source_audio_indexed = 63779
db_calls_indexed = 4809
blocked = 0
would_import = 297
already_present_audio = 0
already_present_db = 0
```

By day:

```text
2026-05-06 = 176
2026-05-07 = 121
```

Interpretation:

- All canonical Mango capture assets are valid.
- None of these 297 recordings exists in `2026-03-09--26` by `phone + start time ±120 sec`.
- None exists in `mango_mvp.db` by the same fuzzy key.
- The full canonical backlog is safe to treat as `would_import` in a future gated import step.

## Proposed import semantics

For each `would_import` item, bridge emits:

- `local_audio_path`: current capture-staging mp3 path.
- `proposed_filename`: legacy-compatible filename candidate.
- `proposed_metadata`: structured mapping for future import.

The future import step should not rely only on filename parsing. It should copy/link audio into a quarantine import folder and provide metadata explicitly:

- `source = mango_api_capture`
- `tenant_id`
- `provider`
- `event_key`
- `provider_call_id`
- `recording_id`
- `started_at_msk`
- `client_phone`
- `manager_ref`
- `direction`
- `duration_sec`
- `checksum_sha256`

## Test gate

Command:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
45 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for local dry-run.

## Next recommended step

Next safe step: `quarantine import dry-run`.

Goal: create a planned import package, still without DB writes:

```text
pipeline_bridge_plan.json
  -> quarantine folder plan
  -> deterministic target filenames
  -> metadata.csv
  -> file copy/link dry-run
  -> validation report
```

Only after manual approval should a later step copy/link files into a real import folder and run a separate DB ingest command.
