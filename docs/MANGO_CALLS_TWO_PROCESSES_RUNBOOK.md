# Mango calls: два процесса

## Контракт

- Process A: Mango API -> локальная загрузка -> локальная SQLite -> ASR -> Resolve -> Analyze -> консистентная backup-копия в drop.
- Process B: готовая drop-копия -> `mango_processed_summary` -> только timeline-staging.
- `sync`, AMO, CRM, Tallanto, prod timeline и `stable_runtime` не используются.
- Resolve/Analyze вызывают Codex через изолированную оболочку без desktop-приложений, плагинов и MCP; используется только подписочная авторизация.
- `--stage-limit` ограничивает один цикл. Полный дренаж обеспечивает worker-loop с `--poll-sec` и `--max-idle-cycles`.
- Launchd запускает по расписанию только `process-a`. После явного `status=ok` его оболочка запускает отдельную demand-only службу `process-b`; при `failed`, `deferred` или `locked` B не стартует. `cycle` остаётся только для ручной совместимости.
- Каждый процесс пишет собственный `state/process_*_status.json`; команда `status` считает свежесть по `data_through`, а не по времени запуска.
- `brand_evidence` (`single`/`both`/`none`) определяется простым поиском маркеров `Фотон`, `УНПК`, `МФТИ` в уже готовом тексте и анализе; модель не вызывается.

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
  "poll_overlap_minutes": 30,
  "api_window_hours": 12,
  "min_free_gib": 40,
  "stage_limit": 20,
  "asr_mode": "mlx_dual",
  "poll_seconds": 10,
  "max_idle_cycles": 30,
  "freshness_max_age_minutes": 90
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

Сначала безопасно отрендерить два plist без установки:

```bash
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config <config.json> \
  --env-file ~/.mango_secrets/mango_office.env \
  --process-a-interval-seconds 1800 \
  --out-dir <папка-проверки>
```

В результате A имеет `StartInterval`, а B не имеет собственного календаря или интервала. После проверки добавить `--install`. Установка сначала загружает demand-only B, затем scheduled A и только после этого выгружает старый общий label; при частичном сбое новые задачи откатываются.

Проверка водяных меток:

```bash
<configured-python> scripts/run_mango_calls_pipeline.py --config <config.json> status
```

`fresh` означает, что дата последнего фактически обработанного звонка моложе заданного порога. Свежий запуск при старых данных остаётся `stale`.

Перед публикацией агрегатного отчёта в `Foton/_daily` встроенный гейт удаляет пути/идентификаторы и блокирует телефон, email или секрет.

## Суточный отчёт для РОПа

Read-only экспорт готовит внутренний XLSX и копии аудиозаписей за последние
полностью завершённые календарные сутки по Москве:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  scripts/export_daily_mango_calls_resolve.py
```

По умолчанию результат создаётся в
`/Users/dmitrijfabarisov/Yandex.Disk.localized/Mango Calls Resolve`:

- аудио лежит в подпапке `Записи разговоров YYYY-MM-DD`;
- XLSX и контрольный JSON лежат в корне;
- готовые звонки и проблемы данных разделены по листам;
- повторный запуск использует уже проверенные копии и блокирует конфликт;
- телефоны не включаются в имена аудиофайлов.

Экспорт проверяет готовый снимок и его manifest, читает обе SQLite только
read-only и не запускает Mango API, ASR, Resolve или Analyze. Строки, которые
есть только в рабочей базе и ещё не завершили обработку, попадают только на
лист проблем. В XLSX содержатся персональные данные, поэтому каталог имеет
режим доступа владельца, а отчёт предназначен только для внутренней проверки.
Автоматическое расписание и удаление старых выгрузок этим шагом не включены.
