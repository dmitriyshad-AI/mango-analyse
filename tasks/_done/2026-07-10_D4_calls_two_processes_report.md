# D4 calls two processes report, 2026-07-10

## Что сделано

- Разрулил перегрузку ASR: остановлен старый тяжёлый запуск с параллельными ASR-стадиями, незавершённые claims возвращены в очередь, готовые результаты сохранены.
- Оркестратор Process A переведён на UI-схему: стадии идут последовательно `transcribe -> backfill-second-asr -> resolve -> analyze`, без одновременного запуска двух ASR worker.
- Одиночный GigaAM-only fallback оставлен запрещённым; основной режим `mlx_dual`.
- Исправлена публикация ready drop SQLite: временная БД после backup приводится к обычному journal-режиму и чистит `.tmp-wal/.tmp-shm`, чтобы `quick_check` не падал на путях с пробелом.
- Process A довёл локальный пакет до drop: 241 звонок, 241 transcription done, 239 second-ASR GigaAM, 164 resolve done, 74 resolve skipped, 3 resolve manual, 238 analysis done, 3 analysis pending/manual.
- Process B влил готовый drop в staging timeline под lock: 238 записей read/accepted, 459 событий/чанков created.

## Ключевые пути

- Working DB: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/working/mango_calls_pipeline.sqlite`
- Ready drop DB: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/drop/mango_calls_ready.sqlite`
- Staging timeline DB: `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore/.codex_local/staging/customer_timeline_staging.sqlite`
- Daily reports: `/Users/dmitrijfabarisov/Claude Projects/Foton/_daily/20260709T23*_process_*_calls.json`
- Main reports:
  - Process A ok: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260709T233309Z_process_a.json`
  - Process B first import: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260709T233319Z_process_b.json`
  - Lock test: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260709T233727Z_process_b.json`
  - Final B repeat: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260709T234111Z_process_b.json`

## Проверки

- Target tests: `41 passed, 1 warning`.
- Full safe collect: `4154 tests collected`, collection errors = 0.
- Process A sequential ASR: одновременно был только один ASR worker; после `transcribe` стартовал только один `backfill-second-asr`.
- Working DB: `PRAGMA quick_check=ok`, `PRAGMA integrity_check=ok`.
- Staging DB: `PRAGMA quick_check=ok`, `PRAGMA integrity_check=ok`.
- Staging source systems for `event_type='mango_call'`: only `mango_processed_summary`.
- Staging dedupe: `COUNT(DISTINCT dedupe_key)=COUNT(*)=75265` for `mango_call`.
- Repeat Process B: all 459 operations became `duplicate`, event count stayed 75265.
- Forced lock test: Process B returned `status=locked`, `stop_reason=timeline_writer_locked`, DB remained ok.
- Technical ASR artifact check: 0 rows with `MTLCompilerService` in `analysis_status='done'`.
- Foton PII sweep on 9 daily reports: phones/email/secrets = 0.

## Счётчики Process B

- Producer rows selected: 238.
- Call type counts: sales_call 119, service_call 68, non_conversation 47, existing_client_progress 4.
- Identity resolution: strong_unique 87, ambiguous 7, unmatched 144.
- Import first run: created 459.
- Import repeats: duplicate 459, no event-count growth.

## Что не делалось

- Не было записей в AMO/CRM/Tallanto.
- Не было записей в production timeline.
- Не трогались `stable_runtime` DB/audio/transcripts.
- Не запускались дополнительные параллельные ASR-процессы после фикса.
- Mango capture в финальных контрольных повторах был `skip_capture`; использовался уже скачанный локальный пакет.

## Остаточные замечания

- Ветка содержит старые изменения базового worktree, поэтому в audit pack широкий `changed_files.txt`; фактический рабочий diff этого шага ограничен 3 файлами: `calls_two_processes.py`, `tests/test_mango_calls_two_processes.py`, `docs/MANGO_CALLS_TWO_PROCESSES_RUNBOOK.md`.
- В отчётах import поле `writes_applied=459` на повторе сопровождается `status_counts.duplicate=459`; фактический контроль по DB подтверждает, что новых строк не добавилось.
