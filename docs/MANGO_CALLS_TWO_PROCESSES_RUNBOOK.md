# Mango calls: два процесса

## Контракт

- Process A: Mango API -> локальная загрузка -> локальная SQLite -> ASR -> Resolve -> Analyze -> консистентная backup-копия в drop.
- Process B: готовая drop-копия -> `mango_processed_summary` -> только timeline-staging.
- `sync`, AMO, CRM, Tallanto, prod timeline и `stable_runtime` не используются.
- Resolve/Analyze вызывают Codex через изолированную оболочку без desktop-приложений, плагинов и MCP; используется только подписочная авторизация.
- `--stage-limit` ограничивает один цикл. Полный дренаж обеспечивает worker-loop с `--poll-sec` и `--max-idle-cycles`.
- Один launchd-trigger вызывает `cycle`: сначала A, затем B. Другой trigger для этой staging DB не устанавливается.

## Конфигурация

Конфигурация хранится вне git, например `.codex_local/mango_calls_two_processes/config.json`:

```json
{
  "pipeline_root": "/absolute/ignored/product_data/mango_calls_two_processes",
  "timeline_db": "/absolute/ignored/staging/customer_timeline_staging.sqlite",
  "timeline_allowed_root": "/absolute/ignored/staging",
  "python_executable": "/usr/bin/python3",
  "codex_binary": "/opt/homebrew/bin/codex",
  "codex_home_root": "/Users/user/.mango_local/mango_calls_pipeline/codex_home",
  "foton_daily_dir": "/absolute/Foton/_daily",
  "bootstrap_since": "2026-07-09T03:00:35+03:00",
  "poll_overlap_minutes": 4320,
  "api_window_hours": 12,
  "min_free_gib": 40,
  "stage_limit": 20,
  "asr_mode": "mlx_dual",
  "poll_seconds": 10,
  "max_idle_cycles": 30
}
```

Как в UI, запускается ровно по одному worker на стадию: Whisper, GigaAM-дозаполнение, Resolve и Analyze. Дублирующих ASR-процессов нет. Пакет `stage_limit=20` сохраняет скорость без повторной загрузки модели на каждый звонок.

Обычный режим идёт по UI-схеме: один worker `transcribe` для Whisper/MLX, затем после его завершения один `backfill-second-asr` для GigaAM, затем отдельные `resolve` и `analyze`. Одиночный ASR-режим отключён: при проблемах Metal/памяти запуск останавливается и разбирается, а не переключается молча на GigaAM-only.

Если `chatgpt.com` не разрешается через DNS, Process A выполняет только локальные стадии ASR и возвращает `deferred/codex_network_unavailable`. Resolve/Analyze не запускаются и их лимит повторов не расходуется; drop публикуется только после следующего успешного полного дренажа.

Секреты Mango хранятся в отдельном env-файле `0600`, не в конфигурации.

## Ручной запуск

```bash
set -a
source ~/.mango_secrets/mango_office.env
set +a
/usr/bin/python3 scripts/run_mango_calls_pipeline.py --config <config.json> process-a
/usr/bin/python3 scripts/run_mango_calls_pipeline.py --config <config.json> process-b
```

## Расписание

После приёмочных прогонов:

```bash
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config <config.json> \
  --env-file ~/.mango_secrets/mango_office.env \
  --interval-seconds 600 \
  --install
```

Перед публикацией агрегатного отчёта в `Foton/_daily` встроенный гейт удаляет пути/идентификаторы и блокирует телефон, email или секрет.
