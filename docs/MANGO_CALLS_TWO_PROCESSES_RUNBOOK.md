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

Read-only экспорт готовит внутренний XLSX и полные TXT-расшифровки за последние
полностью завершённые календарные сутки по Москве:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  scripts/export_daily_mango_calls_resolve.py
```

По умолчанию результат создаётся в `~/Yandex.Disk.localized/Mango Calls Resolve`.
Для автоматического запуска после успешного Process B укажите в env абсолютный
путь `MANGO_CALLS_DAILY_EXPORT_OUT`; без него поведение Process B не меняется.
Пути pipeline, выгрузки и read-only env можно переопределить переменными
`MANGO_CALLS_PIPELINE_ROOT`, `MANGO_CALLS_TALLANTO_EXPORT`,
`MANGO_CALLS_TALLANTO_ENV` и `MANGO_CALLS_MANGO_ENV`:

- расшифровки лежат в неизменяемой подпапке с датой и короткой контрольной суммой;
- соответствующий XLSX и контрольный JSON лежат в корне, JSON указывает текущую версию;
- краткое содержание и полная расшифровка расположены рядом, длинный текст без
  обрезки разбивается на несколько соседних столбцов и сохраняется целиком в TXT;
- ФИО клиента ищется по нормализованному телефону сначала в локальной выгрузке
  Tallanto, затем среди изменённых после выгрузки карточек через read-only API;
- ФИО менеджеров обновляются из текущего read-only справочника Mango API;
- в основной лист и статистику менеджеров попадают только звонки без замечаний,
  с подтверждённым порядком и доказанным соответствием дорожек ролям;
- неизменный повтор использует уже проверенную версию, а изменившийся день
  публикуется новым атомарным поколением; прошлое поколение остаётся для отката;
- телефоны не включаются в имена TXT-файлов.

Экспорт проверяет готовый снимок и его manifest, читает обе SQLite только
read-only, обращается к Mango и Tallanto только за справочными данными и не
запускает ASR, Resolve или Analyze. Строки, которые
есть только в рабочей базе и ещё не завершили обработку, попадают только на
лист проблем. В XLSX содержатся персональные данные, поэтому каталог имеет
режим доступа владельца, а отчёт предназначен только для внутренней проверки.
Удаление старых выгрузок не выполняется.

### Google-таблица на M1

Если одновременно заданы `MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID` и
`GOOGLE_APPLICATION_CREDENTIALS`, Process B после локального v3-XLSX загружает
его в указанную папку Drive и преобразует в нативную Google-таблицу. Одна пара
«день + `content_sha256`» создаётся только один раз; изменённый день получает
новое неизменяемое поколение. В названии есть время публикации по Москве:
при нескольких поколениях текущим считается самое позднее. После загрузки
служба скачивает таблицу обратно как XLSX и сверяет листы, значения, типы ячеек,
формулы и ссылки. Составная загрузка ограничена 5 МБ:
больший отчёт остаётся локально и завершает Google-шаг ошибкой без потери файла.

Файл service account должен находиться только в `~/.mango_secrets/` с режимом
`0600`. Репозиторий и Яндекс Диск для него запрещены. Папку Drive нужно заранее
дать service account право редактировать; публичная или доменная папка
блокируется до загрузки. До успешной обратной сверки файл имеет имя
`ПРОВЕРКА — НЕ ИСПОЛЬЗОВАТЬ`; при ошибке он удаляется либо остаётся с этим
карантинным именем, но не выдаётся за готовый. Относительные ссылки на TXT
заменяются честным текстовым путём на Яндекс Диске, потому что сами TXT в Google
не копируются; полный диалог остаётся в соседних столбцах без обрезки. Без обоих
параметров Google Drive не вызывается. Ручной dry-run без сети:

```bash
python3 scripts/publish_daily_mango_calls_google.py \
  --report-root "$MANGO_CALLS_DAILY_EXPORT_OUT" --day YYYY-MM-DD
```

Механизм использует официальное преобразование Excel в Google Sheets при
создании файла через Google Drive API; существующая native-таблица не
перезаписывается медиа-загрузкой.

Последовательный диалог публикуется только при наличии сохранённых временных
меток и ролей. Если старый ASR-артефакт содержит лишь два полных ролевых блока,
экспорт не придумывает очередность: добавляет предупреждение, сохраняет весь
текст без сокращений и относит строку к проблемам данных.
