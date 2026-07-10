# Контрольная точка: два процесса звонков

Дата: 2026-07-09

## Где продолжать

- Worktree: `/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes`
- Ветка: `codex/calls-two-processes`
- ТЗ: `tasks/_running/2026-07-09_CODEX_D4_TZ_calls_two_processes.md`
- Локальная конфигурация: `.codex_local/mango_calls_two_processes/config.json` (в git не входит)
- Рабочие данные: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/` (в git не входят)

## Сделано

- Реализованы Process A и Process B, единый цикл и установщик одного launchd-задания.
- Process A: Mango discovery/download, manifest/dedupe, диск-гейт, fcntl-лок, локальная call DB, worker-дренаж, atomic SQLite drop, JSON-отчёты.
- Process B: producer `mango_processed_summary`, импорт только в timeline-staging, graceful `locked`, source-system и integrity гейты.
- Из Mango скачана 241 новая запись; повторно старые пакеты не скачиваются.
- Подготовлен изолированный Codex CLI runtime без desktop MCP/plugins.
- Целевые тесты: `35 passed`.

## Важный инцидент и исправление

- Ошибочно запускались несколько одинаковых ASR-worker. Это перегрузило Metal и дало технические ошибки `MTLCompilerService`.
- Все дополнительные процессы остановлены. Локальная call DB несколько раз сохранена через SQLite backup.
- Только технически затронутые строки возвращены в очередь; готовые результаты сохранены.
- Аварийный режим `gigaam_fallback` удалён из кода и локальной конфигурации.
- Обязательный контракт дальше: как в UI, ровно по одному worker на стадию `transcribe`, `backfill-second-asr`, `resolve`, `analyze`. Никогда не запускать несколько одинаковых ASR-worker.

## Текущее состояние данных

- Активных worker-процессов: 0.
- Локальная call DB: 241 строка.
- Transcribe: 47 `done`, 194 `pending`.
- Resolve: 30 `done`, 1 `manual`, 9 `skipped`, 201 `pending`.
- Analyze: 2 `done`, 239 `pending`.
- Активных lease/claim: 0.
- `PRAGMA quick_check`: `ok`.
- Валидный готовый drop ещё не опубликован.
- Process B на этом пакете ещё не запускался.
- В prod timeline, stable_runtime, AMO, CRM и Tallanto записей не было.

## Следующий безопасный шаг

1. Ещё раз сверить `on_parallel_pipeline_start`, `_env_for_stage_worker` в `src/mango_mvp/gui.py` и текущий `run_parallel_pipeline_workers`.
2. Запустить Process A с защищённым env-файлом и `--skip-capture`, не создавая дополнительных worker вручную.
3. Сразу проверить: один Whisper, один GigaAM, один Resolve, один Analyze; память без давления.
4. Дождаться полного дренажа и публикации drop.
5. Запустить Process B на staging, затем 3-5 повторов A -> B без дублей.
6. Сделать forced-lock тест B, `quick_check`/`integrity_check`, PII sweep, полный pytest и финальный отчёт.

## Стоп-правила

- Не запускать несколько одинаковых ASR-процессов.
- Не писать в prod timeline, stable_runtime, AMO, CRM или Tallanto.
- Не устанавливать расписание до полной приёмки A -> B.
- При Metal/память-ошибке остановить запуск, сохранить SQLite backup и расследовать причину; не переключаться молча на один ASR-движок.
