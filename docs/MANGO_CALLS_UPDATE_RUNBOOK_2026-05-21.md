# Mango Calls Update Runbook

Дата: 2026-05-21

Назначение: быстрый повторяемый процесс обновления базы звонков через Mango API: найти новые записи, скачать недостающие аудио, подготовить ASR, выполнить ASR, затем Resolve+Analyze и пересобрать downstream-слои.

## 0. Жёсткие правила безопасности

Без отдельного явного подтверждения Дмитрия нельзя:

- скачивать записи из Mango;
- запускать ASR;
- запускать Resolve+Analyze;
- менять `stable_runtime` DB/audio/transcripts;
- переключать `CURRENT_RUNTIME.json`;
- писать в AMO/CRM/Tallanto;
- удалять или перемещать старые аудио/runtime-артефакты.

Разрешено без отдельного подтверждения только read-only изучение файлов и подготовка плана.

## 1. Быстрый вызов процесса

Если Дмитрий пишет:

```text
Запусти Mango update по runbook после <дата>
```

то Codex должен открыть этот файл и идти по стадиям:

1. read-only сверка Mango API;
2. отчёт “что есть в Mango, чего нет у нас”;
3. отдельное подтверждение на скачивание;
4. скачивание аудио;
5. подготовка ASR-only batch;
6. отдельное подтверждение на ASR;
7. ASR;
8. отдельное подтверждение на Resolve+Analyze;
9. Resolve+Analyze;
10. отдельное подтверждение на rebuild runtime/downstream;
11. canonical/downstream rebuild и audit pack.

## 2. Текущий источник истины перед запуском

Перед каждым запуском читать:

- `AGENTS.md`
- `docs/CURRENT_STATE.md`
- `docs/DECISIONS_LOG.md`
- `docs/ROADMAP.md`
- `docs/RUNBOOK.md`
- `stable_runtime/CURRENT_RUNTIME.json`
- `stable_runtime/CANONICAL_EXPORT.txt`

На дату создания этого runbook текущий принятый runtime:

- active export: `stable_runtime/sales_master_export_20260516_after_mango_update_v1`
- canonical DB: `stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db`
- actionable calls: `65 100`
- missing ASR: `0`
- missing Resolve+Analyze: `0`

Отдельно помнить: слой `2026-05-17` мог быть обработан дальше, чем зафиксировано в `CURRENT_RUNTIME`, но не считается accepted runtime до отдельного rebuild/gates.

## 3. Стадия A: read-only сверка Mango API

Цель: понять, какие звонки есть в Mango после нужной даты и какие из них отсутствуют у нас.

Пример для периода после 12 мая 2026:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_shadow_poll.py \
  --tenant foton \
  --since "2026-05-12T00:00:00+03:00" \
  --until "<CURRENT_ISO_DATETIME_WITH_TZ>" \
  --raw-payload-jsonl product_data/mango_audio_update_<RUN_ID>/mango_shadow_poll_raw.jsonl \
  --out product_data/mango_audio_update_<RUN_ID>/mango_shadow_poll.json
```

Эта стадия не скачивает аудио, не запускает ASR/R+A и не пишет в CRM.

Выход стадии:

- raw Mango payload JSONL;
- shadow poll summary JSON;
- список кандидатов на скачивание/исключение.

## 4. Стадия B: построить план скачивания

Цель: превратить найденные записи в controlled download plan.

Основной безопасный путь:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_recording_capture_plan.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  build \
  --manifest <capture_plan_manifest.jsonl> \
  --out <capture_plan_report.json>
```

Использовать актуальные пути конкретного run-а. Не использовать старый `scripts/mango_office_download_recordings.py`, если нет отдельной причины: он legacy и опаснее guarded downloader.

Проверить в отчёте:

- сколько `PLAN_DOWNLOAD_DRY_RUN`;
- сколько `SKIP_DUPLICATE_RECORDING`;
- сколько `SKIP_EXISTING_FILE`;
- сколько `BLOCK_MISSING_RECORDING_REF`;
- нет ли target paths вне разрешённой папки.

## 5. Стадия C: скачать недостающие записи

Требует отдельного подтверждения Дмитрия.

Подтверждение должно быть явным, например:

```text
Подтверждаю скачивание недостающих Mango recordings для run <RUN_ID>
```

Команда-шаблон:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  run \
  --source-plan <capture_plan_manifest.jsonl> \
  --recordings-dir <recordings_dir> \
  --download-manifest <recording_download_manifest.jsonl> \
  --out <recording_download_report.json> \
  --execute \
  --sleep-sec 1.5
```

После скачивания обязательно запустить audit download manifest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_recording_capture_download.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  audit \
  --download-manifest <recording_download_manifest.jsonl> \
  --recordings-dir <recordings_dir> \
  --out <recording_download_audit.json>
```

Критерий прохода:

- `validation_ok=true`;
- `failed_latest_events=0`;
- `missing_files=0`;
- `zero_size_files=0`;
- `checksum_mismatches=0`.

## 6. Стадия D: подготовить ASR-only batch

Цель: собрать скачанные аудио в batch-папку с нормальными именами, metadata и launcher scripts.

Скрипт:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_mango_new_calls_asr_ra_batch.py \
  --plan-csv <needs_asr_ra_plan.csv> \
  --out-dir product_data/mango_new_calls_<RUN_ID>_asr_ra
```

Что важно:

- ФИО менеджеров в имени файла должны быть человекочитаемыми;
- для исправления старой битой кодировки есть `src/mango_mvp/utils/filename_repair.py`;
- audio-файлы не должны дублироваться;
- batch должен содержать `metadata.csv`, `audio/`, `transcripts/`, `logs/`, launcher scripts.

## 7. Стадия E: ASR

Требует отдельного подтверждения Дмитрия.

Подтверждение:

```text
Подтверждаю запуск ASR для batch <BATCH_ROOT>
```

Правило нагрузки:

- запускать один ASR worker, если нет отдельной причины;
- не запускать несколько ASR workers одновременно: Whisper/MLX и GigaAM сами активно используют ресурсы.

Обычно запускать prepared script из batch-папки:

```bash
<BATCH_ROOT>/run_02_transcribe_*.sh
```

Или через CLI:

```bash
DATABASE_URL="sqlite:///<BATCH_DB>" \
TRANSCRIPT_EXPORT_DIR="<BATCH_ROOT>/transcripts" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m mango_mvp.cli worker \
  --stages "transcribe,backfill-second-asr" \
  --stage-limit 1 \
  --poll-sec 5 \
  --max-idle-cycles 5
```

После ASR проверить:

```bash
DATABASE_URL="sqlite:///<BATCH_DB>" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m mango_mvp.cli stats
```

Критерий:

- transcribe done по всем batch-звонкам;
- нет failed/dead хвостов без решения.

## 8. Стадия F: Resolve + Analyze

Требует отдельного подтверждения Дмитрия.

Подтверждение:

```text
Подтверждаю запуск Resolve+Analyze для batch <BATCH_ROOT>
```

Правило нагрузки:

- безопасный режим: `1 Resolve + 3 Analyze`;
- не использовать максимум `2+6`, если нет отдельной причины.

Команда-шаблон:

```bash
DATABASE_URL="sqlite:///<BATCH_DB>" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m mango_mvp.cli worker \
  --stages "resolve" \
  --stage-limit 1 \
  --poll-sec 5 \
  --max-idle-cycles 60
```

Параллельно до 3 Analyze workers:

```bash
DATABASE_URL="sqlite:///<BATCH_DB>" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m mango_mvp.cli worker \
  --stages "analyze" \
  --stage-limit 1 \
  --poll-sec 5 \
  --max-idle-cycles 60
```

Критерий:

- Resolve done/skipped по всем допустимым строкам;
- Analyze done по всем допустимым строкам;
- manual/dead хвосты явно перечислены и не скрыты.

## 9. Стадия G: интеграция в accepted runtime

Требует отдельного подтверждения Дмитрия.

Не добавлять новые строки прямо в старую accepted DB. Правильный путь: новая версионированная runtime-база.

Обычно нужны:

1. canonical master rebuild;
2. insight readiness / phone-chain rebuild;
3. sales export rebuild;
4. CRM/writeback quality gate;
5. AMO queue rebuild;
6. deal-aware rebuild, если эти звонки должны участвовать в сделках;
7. Telegram/context layer rebuild, если эти звонки должны участвовать в ответах.

Критерии перед переключением `CURRENT_RUNTIME.json`:

- canonical validation ok;
- missing ASR = 0 для accepted actionable scope;
- missing R+A = 0 для accepted actionable scope;
- CRM quality gate green;
- AMO queue классифицирует все строки;
- audit pack создан;
- старый runtime не удалён.

## 10. Что отдавать Дмитрию после каждого этапа

После read-only сверки:

- сколько звонков найдено в Mango;
- сколько уже есть у нас;
- сколько надо скачать;
- сколько без recording_ref;
- за какой период проверяли.

После скачивания:

- downloaded / skipped / failed;
- путь к recordings dir;
- путь к manifest;
- есть ли checksum/missing/zero-size ошибки.

После ASR:

- всего batch rows;
- transcribe done/pending/failed/dead;
- где лежат transcripts;
- сколько осталось хвостов.

После R+A:

- resolve done/skipped/manual/failed;
- analyze done/pending/failed;
- список manual хвостов;
- рекомендация: rebuild сейчас или сначала разобрать хвосты.

После rebuild:

- новый canonical root;
- новый active export candidate;
- gate summaries;
- что можно/нельзя использовать дальше.

## 11. Запреты на автоматическое продолжение

Codex должен остановиться и попросить подтверждение перед каждым переходом:

- read-only сверка -> скачивание;
- скачивание -> ASR;
- ASR -> Resolve+Analyze;
- Resolve+Analyze -> accepted runtime rebuild;
- rebuild -> переключение runtime;
- любой этап -> AMO/CRM/Tallanto write.

## 12. Ближайший рекомендуемый запуск

Для обновления после 12 мая 2026 года начать с read-only сверки:

```text
Проверить Mango API с 2026-05-12T00:00:00+03:00 по текущий момент и подготовить отчёт, ничего не скачивать.
```

После отчёта отдельно решить, скачивать ли недостающие записи и запускать ли ASR.
