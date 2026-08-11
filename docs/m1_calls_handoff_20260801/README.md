# Передача конвейера звонков на M1

Актуализировано: 2026-08-11.

Это главная инструкция пакета. Она готовит M1, но не доказывает, что служба уже
работает. Реальные аудио, базы, ASR, Resolve, Analyze и запуск `launchd` требуют
отдельных подтверждений владельца по фазам.

## Итоговая схема

- M1 отдельно выполняет лёгкий capture, а затем один тяжёлый pipeline:
  Whisper → GigaAM → Resolve → Analyze → sealed ready DB.
- Demand-only Process B вызывается тяжёлым pipeline и пишет только в
  отдельную локальную Customer Timeline staging на M1.
- Phase A создаёт только owner-only local safe-plan и суточные пакеты.
  Google, Яндекс.Диск, production Timeline и основной Mac не изменяются.
- Plist только отрисовываются для проверки; launchd не устанавливается и не
  запускается.

Постояный `launchd` остаётся будущей фазой после лестницы пилота и
отдельного решения владельца.

## Канонические документы

Читать строго в таком порядке:

1. этот `README.md`;
2. `tasks/_running/2026-08-11_TZ_m1_calls_service_fast_value.md`;
3. `docs/MANGO_CALLS_TWO_PROCESSES_RUNBOOK.md`;
4. `tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md`;
5. `M1_FAST_SERVICE_CODEX_PROMPT_20260811.md` для запуска новой задачи Codex на M1;
6. `docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md` — только будущие
   selective transfer/relocation после отдельного допуска Phase C.

Старый внешний `CANONICAL_GIT_SHA.txt` от 2026-08-01 не является актуальным.
Новый `CUTOVER_CODE_SHA.txt` создаётся только после завершения всех правок,
аудита, слияния в `main` и явной фиксации версии владельцем.

## Что идёт через Git

- код получения и обработки звонков;
- установщик зависимостей и службы;
- примеры конфигурации без значений секретов;
- тесты, инструкции, план переноса и ТЗ.

Через Git, чат, почту, audit pack и папку передачи не идут:

- токены, ключи, `auth.json` и Google service account;
- аудио, SQLite, расшифровки и выгрузки Tallanto;
- файлы с телефонами, ФИО и другими персональными данными.

Рабочие данные и секреты передаются только напрямую по SSH/`scp`/`rsync` после
отдельного подтверждения. Яндекс Диск используется только для разрешённых
внутренних XLSX/TXT и необязательного архива навыков без токенов.

## Что может положить другой диалог в папку передачи

Папка передачи является указателем, а не второй копией проекта. Достаточно:

- `START_HERE.txt` со ссылкой на эту инструкцию;
- `CUTOVER_CODE_SHA.txt`, созданного после финального слияния;
- копии `M1_CODEX_PROMPT.md`;
- архива пользовательских skills без токенов и его SHA-256;
- обезличенного `TRANSFER_MANIFEST.json` с именами, размерами и SHA разрешённых
  файлов.

Если SHA папки передачи не равен `git rev-parse origin/main`, работу остановить.
Не пытаться «починить» расхождение выбором старой ветки.

## Подготовка M1 без запуска обработки

### 1. Проверить пользователя и путь

Фактический пользователь M1 — `dmitriy`. Репозиторий и все локальные пути
строятся от `$HOME`; ожидаемый каталог репозитория:

```text
$HOME/Projects/Mango analyse
```

Старый корень source Mac и новый корень M1 передаются relocation-скрипту как
разные явные параметры. Простая строковая замена внутри SQLite запрещена.

### 2. Получить код

```bash
set -euo pipefail
mkdir -p "$HOME/Projects"
git clone git@github.com:dmitriyshad-AI/mango-analyse.git \
  "$HOME/Projects/Mango analyse"
cd "$HOME/Projects/Mango analyse"
git fetch origin
git switch main
git pull --ff-only origin main
test -z "$(git status --porcelain)"
git rev-parse HEAD
```

Полученный SHA должен совпасть с отдельно подтверждённым `CUTOVER_CODE_SHA`.

### 3. Установить зависимости

Сначала посмотреть план. Скрипт не запускает обработку и службы:

```bash
set -euo pipefail
scripts/bootstrap_m1_mango_calls.sh plan
CONFIRM_M1_PACKAGE_INSTALL=INSTALL_M1_MANGO_CALLS_PACKAGES \
  scripts/bootstrap_m1_mango_calls.sh install
```

Штатный проверенный Codex CLI пока закреплён на `0.142.3`. Resolve использует
`gpt-5.4`, Analyze - `gpt-5.4-mini`. Версии моделей и CLI не менять во время
переноса без отдельного сравнительного теста.

### 4. Войти в Codex

```bash
set -euo pipefail
codex login
codex login status
```

Не копировать `auth.json`. Resolve и Analyze работают в отдельном профиле без
skills, plugins и MCP. Пользовательские skills нужны только Codex-разработчику.

### 4.1 Проверить готовность ровно одного звонка

Для controlled-1 используется отдельная копия runtime-конфига. Обычный
service-конфиг сохраняет `"processing_scope": "service"` и
`"stage_limit": 20`. После переноса рабочей БД и выбора одного уже скачанного
звонка создать owner-only allowlist без API и моделей:

```bash
set -euo pipefail
mkdir -p "$HOME/.mango_local/mango_calls_controlled_one"
chmod 700 "$HOME/.mango_local/mango_calls_controlled_one"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "$HOME/.mango_local/mango_calls_runtime/venv/bin/python" \
  scripts/create_m1_calls_controlled_allowlist.py \
  --config "$HOME/.mango_local/mango_calls_two_processes/config.json" \
  --source-call-id '<EXACT_SOURCE_CALL_ID>' \
  --out "$HOME/.mango_local/mango_calls_controlled_one/allowlist.json"
```

Скрипт откажет до записи, если SHA кода не совпадает, worktree грязный,
фактический owner-only `host_id` другой, БД неавторитетна,
`source_call_id` отсутствует/дублируется либо исходное аудио отсутствует, пусто,
является символической ссылкой или лежит вне рабочего audio-каталога. Штатный
hardlink, созданный `hardlink_or_copy`, разрешён и проверяется по inode, SHA и
размеру; оба ASR читают отдельную приватную копию запуска.
После проверки строки и аудио этот же скрипт read-only подтверждает lineage
перенесённого cursor: повторно требует свежий shutdown-proof исходного Mac,
точный SHA cursor и неизменный cutover manifest. Общий service-маркер
`cutover_cursor_lineage.json` он не создаёт, поэтому обычные capture/Process A
остаются STOP до отдельного разрешения cutover. Mango API, модели и обработку
звонка скрипт не запускает. На время проверки и записи allowlist он удерживает
локальные pipeline и capture lock; занятый lock, stale или несовпадающее
доказательство означает STOP до создания allowlist.
Полученный SHA записать в отдельный `config.controlled-one.json` вместе с:

```json
{
  "processing_scope": "controlled_1",
  "stage_limit": 1,
  "controlled_call_allowlist_path": "<HOME>/.mango_local/mango_calls_controlled_one/allowlist.json",
  "controlled_call_allowlist_sha256": "<ALLOWLIST_SHA256>"
}
```

Этот конфиг запрещает capture, ingest, `run-all`, Process A/B, cycle, sync и
publication через оркестратор; самостоятельные publication-скрипты остаются
за своими confirmation-гейтами и в этой фазе не запускаются. Прямые тяжёлые
CLI-команды в controlled scope запрещены. Каждый отдельный
Whisper/GigaAM/Resolve/Analyze worker получает только свежий короткоживущий
owner-only stage-ticket родительской команды, заново проверяет manifest, SHA,
code/tenant/host binding, фактический owner-only `host_id`, PID и удерживаемый
pipeline lock, точные production-провайдеры и единственную строку БД. Ticket
защищает от случайного прямого CLI и устаревшего запуска, но не является
криптографической защитой от произвольного процесса того же пользователя
`dmitriy`: same-UID является локальной границей доверия. Allowlist также связан
с ID строки, SHA и размером аудио. `stage_limit=1` сам по себе изоляцией не
считается.

Если после аварийного завершения осталась приватная копия в
`$HOME/.mango_local/.../state/controlled_runs/`, следующий controlled-one
завершится STOP до ASR. Не удалять её автоматически: сначала подтвердить, что
worker не запущен, сверить owner/права и получить решение владельца на очистку.
Если стадии уже завершились, но удалить каталог копии не удалось, команда
сохраняет полный отчёт `before/stages/after`, помечает его `status=failed` и
`pilot_transition_proven=false`, а остаток продолжает блокировать повтор. Так
результат не теряется, но ошибка уборки не превращается в успешный пилот.

Только когда на M1 не идёт тяжёлая сборка Customer Timeline, разрешена
синтетическая проверка моделей без аудио и текста клиентов:

```bash
set -euo pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "$HOME/.mango_local/mango_calls_runtime/venv/bin/python" \
  scripts/probe_m1_calls_access.py \
  --config "$HOME/.mango_local/mango_calls_two_processes/config.controlled-one.json" \
  --run-offline-model-probes \
  --run-codex-model-probes \
  --readiness-target controlled-1
```

`controlled-1` не требует замера десяти звонков. Обычный запуск без
`--readiness-target` остаётся строгим `service` и вернёт успех только после
отдельного доказательства capacity. В отчёте `requested`, `attempted` и
успешные синтетические вызовы разделены; реальные Resolve/Analyze, аудио и
внешние записи эта команда не выполняет.

`service_machine_preflight=OK` означает только готовность машины и не разрешает
запуск службы. `production_service_readiness` и старый `host_readiness`
остаются `STOP`, пока отдельный owner-reviewed controlled-one evidence не
свяжет реальный результат, ручную проверку и идемпотентный повтор.

После отдельного разрешения на один реальный звонок единственная допустимая
команда тяжёлого пилота будет:

```bash
set -euo pipefail
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "$HOME/.mango_local/mango_calls_runtime/venv/bin/python" \
  scripts/run_mango_calls_pipeline.py \
  --config "$HOME/.mango_local/mango_calls_two_processes/config.controlled-one.json" \
  controlled-one
```

Сейчас эту команду не выполнять. Она не делает capture, не запускает Process B
и не публикует ничего наружу. Отчёт сравнивает digest всех нецелевых строк до и
после. Первый новый полный проход получает
`execution_class=transitioned_to_ready`; уже готовая строка —
`idempotent_noop` и не выдаётся за первый пилот. Отдельный повтор по готовому
звонку обязан показать по четыре `processed=0`. В обоих отчётах
`controlled_1_human_pass=false`, `business_pass=false` и `runtime_pass=false`
до ручной сверки аудио и отдельного гейта.
`pilot_transition_proven=true` дополнительно требует фактические runtime-
квитанции MLX Whisper, успешной очистки MLX-кэша и GigaAM; cache reuse или
отсутствующий финальный JSON worker не выдаются за свежий пилот. Controlled-
артефакты пишутся в изолированный каталог по хэшу `source_call_id`, поэтому
совпадающие имена файлов других звонков не перезаписываются.

### 5. Подготовить локальные файлы

- `~/.mango_secrets/` - каталог `0700`;
- настоящий `~/.mango_local/` текущего пользователя - каталог `0700`, не
  символическая ссылка; runbook проверяет владельца до исправления режима;
- `mango_calls_m1_worker.env`, `mango_office.env`,
  `tallanto_readonly.env` - строго `0600`;
- `~/.mango_local/tallanto/Contacts_current.csv` - `0600` и свежая дата
  содержимого, указанная в `MANGO_CALLS_TALLANTO_SNAPSHOT_AS_OF`;
- шаблон `config.m1.example.json` только просмотреть, но финальные
  `pipeline_root` и `config.json` до первичной передачи не создавать: runbook
  атомарно создаст runtime, затем `config.json` с режимом `0600`;
- итоговый `pipeline_root` будет `0700`, только внутри `~/.mango_local`, вне
  Git, Яндекс Диска и `~/Library/CloudStorage`.

Значения `<HOME>`, `<CUTOVER_BOOTSTRAP_SINCE_ISO8601>`,
`<EXPECTED_CODE_SHA>`, `<OWNER_EMAIL>`, `<ROP_EMAIL>` и
`<CURRENT_TALLANTO_SNAPSHOT_ISO8601>` обязательно заменить. `<HOME>` всегда
берётся из фактического `$HOME` M1; имя пользователя в шаблонах не зашивается.
Старый снимок Tallanto от 2026-06-20 не использовать как текущий.

### 6. Будущая проверка Яндекс Диска — не выполнять в Phase A

Этот раздел требует отдельного допуска внешней публикации. В Phase A полный
пакет остаётся под `$HOME/.mango_local`; клиент Яндекс Диска, marker и
круговой тест не нужны. После отдельного допуска сначала убедиться в интерфейсе,
что папка синхронизируется, и только затем создать локальный маркер:

```bash
set -euo pipefail
YANDEX="$HOME/Yandex.Disk.localized/Mango Calls Resolve"
test -d "$YANDEX" && test -w "$YANDEX" && test ! -L "$YANDEX"
printf 'mango-calls-yandex-v1\n' > "$YANDEX/.mango_calls_yandex_target"
chmod 600 "$YANDEX/.mango_calls_yandex_target"
```

Маркер защищает от случайного создания обычной локальной папки, но не доказывает
облачную доставку. Перед пилотом нужен ручной круговой тест: создать безопасный
файл, увидеть его с другого устройства и удалить только этот тестовый файл.
Обезличенный результат записать в owner-only `phase1_yandex_roundtrip.json`.

### 7. Проверить хост после первичной передачи

Этот шаг выполняется только после раздела «Первичная передача рабочего
состояния» runbook: к этому моменту каталог runtime и финальный `config.json`
уже существуют. До передачи разрешены только `plan`, установка зависимостей и
проверка локальных файлов вручную.

```bash
set -euo pipefail
MANGO_CALLS_CONFIG="$HOME/.mango_local/mango_calls_two_processes/config.json" \
MANGO_CALLS_ENV_FILE="$HOME/.mango_secrets/mango_calls_m1_worker.env" \
  scripts/bootstrap_m1_mango_calls.sh check
```

Проверка выводит только `true/false`. Блокирующие поля перечислены в итоговом
условии bootstrap: платформа, конфигурация и env, импорты, ffmpeg/ffprobe,
Mango/Codex/Tallanto, свежий Tallanto-файл, допустимая Google-конфигурация,
права каталогов, безопасный runtime-путь, место на
диске, чистый SHA, отсутствие конфликтующих служб и lock. Поля
`developer_profile_ready`, `google_publish_enabled` и
`external_publication_ready`, `network_access_verified` справочные и не
блокируют локальную Phase A. Яндекс-папка в bootstrap Phase A
не проверяется. `network_access_verified=false` означает,
что реальные read-only доступы ещё не доказаны; их проверяют отдельными
микропробами без скачивания партии и без печати персональных данных.

## Текущий честный статус

- Базовый код fast-service, безопасный render и локальная публикационная
  подготовка существуют.
- Phase A plist не вызывает Google/Яндекс. Текущий Google safe-plan и суточные
  final/incomplete пакеты создаются только локально; внешний execute требует
  отдельного допуска и exact SHA короткоживущего плана.
- Selective transfer переносит все неизвестные/незавершённые/multi-аудио и
  исключает только строго доказанное готовое историческое аудио.
- До live-cutover M1 должен выполнить ТЗ готовности, закрыть указанные в нём
  гейты полноты и пройти независимый аудит.
- После этого выполняются ручной контрольный цикл, отдельное решение владельца
  о запуске службы и семь последовательных зелёных суток.

До завершения семисуточного пилота допустимые статусы:

```text
formal_pass: true/false
semantic_pass: true/false
data_pass: true/false
runtime_pass: false
pilot_ready: false
production_ready: false
```
