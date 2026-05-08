# Mango capture staging audit

Дата: 2026-05-07

Цель: зафиксировать первый idempotent capture staging слой для Mango API без записи в runtime-БД, без ASR/R+A и без CRM writeback.

## Что добавлено

Новые productization-компоненты:

- `src/mango_mvp/productization/capture_staging.py`
- `src/mango_mvp/productization/mango_recordings.py`
- `scripts/mango_office_capture_stage.py`
- `scripts/mango_office_capture_audit.py`

Назначение:

- poll/read Mango events;
- stage events into canonical JSONL manifest;
- download or reuse recordings;
- validate local audio asset;
- compute checksum;
- record duration/codec/channels/sample_rate;
- skip already staged events;
- link duplicate recording rows to canonical asset;
- audit missing/zero/checksum/duration/unreferenced files.

## Safety boundaries

Этот слой не делает:

- запись в `stable_runtime`;
- запись в текущую SQLite runtime-БД;
- ASR/R+A;
- AMO/Tallanto writeback;
- изменение batch/start/run-ui scripts.

## Canonical manifest

Файл:

```text
_local_archive_mango_api_downloads_20260507/capture_manifest.jsonl
```

Schema:

```text
capture_manifest_v1
```

Основные поля:

- `tenant_id`
- `provider`
- `event_key`
- `provider_call_id`
- `recording_id`
- `started_at`
- `ended_at`
- `direction`
- `client_phone`
- `manager_ref`
- `status`
- `local_audio_path`
- `size_bytes`
- `checksum_sha256`
- `duration_sec`
- `codec_name`
- `channels`
- `sample_rate`

## Реальный staging по скачанным Mango записям

Вход:

```text
/tmp/mango_missing_vs_audio_20260507_fuzzy.json
```

Команда:

```zsh
PYTHONPATH=src python3 scripts/mango_office_capture_stage.py \
  --from-report /tmp/mango_missing_vs_audio_20260507_fuzzy.json \
  --out-root _local_archive_mango_api_downloads_20260507 \
  --out-dir _local_archive_mango_api_downloads_20260507/recordings \
  --manifest _local_archive_mango_api_downloads_20260507/capture_manifest.jsonl \
  --audit-out _local_archive_mango_api_downloads_20260507/capture_audit.json \
  --sleep-sec 0
```

Результат первого staging pass:

```text
total_events = 323
reused_existing_file = 297
already_manifested = 26
downloaded = 0
failed = 0
manifest_rows = 297
```

Почему `297`, а не `323`: в Mango report есть repeated rows with the same `provider_call_id`. Canonical manifest хранит уникальный event-level record. Повторные строки не создают новые manifest entries.

## Audit result

Файл:

```text
_local_archive_mango_api_downloads_20260507/capture_audit.json
```

Результат:

```text
latest_unique_events = 297
downloaded_latest_events = 297
missing_files = 0
zero_size_files = 0
checksum_missing = 0
duration_missing = 0
recordings_dir_mp3_files = 311
recordings_dir_total_mb = 198.53
unreferenced_audio_files = 14
```

Вывод:

- Canonical manifest целостный.
- Все 297 canonical assets имеют файл, checksum и duration.
- Нулевых файлов нет.
- 14 unreferenced mp3 появились из первого downloader pass до canonical dedupe; их не удалять автоматически. Это forensic/artifact spillover, не blocking error.

## Idempotency check

Повторный запуск той же команды:

```text
manifest lines before = 297
manifest lines after = 297
total_events = 323
already_manifested = 323
failed = 0
```

Вывод: повторный запуск не добавляет дублей и не скачивает уже staged события.

## Live dry-run poll check

Команда:

```zsh
rm -rf /tmp/mango_capture_stage_live_dry_run
PYTHONPATH=src python3 scripts/mango_office_capture_stage.py \
  --tenant foton \
  --hours 0.25 \
  --dry-run \
  --out-root /tmp/mango_capture_stage_live_dry_run \
  --sleep-sec 0
```

Результат:

```text
total_events = 8
dry_run_download = 5
skipped_no_recording = 3
failed = 0
```

Вывод: новый product path `poll Mango -> normalize -> manifest stage -> audit` работает без скачивания.

## Test gate

Команда:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Результат:

```text
37 passed, 1 warning
```

Warning: локальный Python/urllib3 сообщает о LibreSSL вместо OpenSSL. Это не блокирует локальный POC, но для client/server appliance нужен управляемый Python runtime.

## Next recommended step

Следующий безопасный SaaS-шаг: `pipeline bridge dry-run`.

Цель: превратить canonical manifest entries в список потенциальных ingest candidates для текущего pipeline, но пока без записи в runtime DB.

Acceptance:

- читает `capture_manifest.jsonl`;
- выбирает только `status=downloaded`;
- проверяет, что файл существует и checksum совпадает;
- строит предполагаемое legacy filename / metadata mapping;
- сравнивает с `2026-03-09--26` и текущей DB только read-only;
- выводит dry-run import plan: `would_import`, `already_present`, `blocked`;
- не запускает ASR/R+A и не пишет в DB.
