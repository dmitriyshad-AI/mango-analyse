# Mango Calls на M1: быстрый последовательный контур

## Статус и границы

Код поддерживает лёгкий захват каждые 15 минут и один тяжёлый координатор
каждые 30 минут. Тяжёлые стадии идут строго последовательно:

1. Whisper `large-v3`;
2. освобождение только свободного MLX-кэша;
3. GigaAM `v2_rnnt` на CPU;
4. Resolve;
5. Analyze;
6. sealed ready-поколение и demand-only Customer Timeline staging.

Phase A разрешает только синтетические тесты и локальные owner-only артефакты.
Она не разрешает реальный Mango capture, ASR реальных звонков, Resolve/Analyze,
установку launchd, cutover, запись в Google/Яндекс/CRM/AMO и изменение
production Customer Timeline или `stable_runtime`.

## Состояние и идемпотентность

Источники истины, по порядку:

- append-only `capture/capture_manifest.jsonl`;
- `working/mango_calls_pipeline.sqlite`;
- sealed `drop/mango_calls_ready.sqlite` и его manifest;
- локальные журналы публикации.

Папка или перемещение аудио не являются статусом. Один `source_call_id` уже в
working SQLite не ingest-ится второй раз. Пропустить отсутствующее историческое
аудио разрешено только для уникальной строго готовой строки, одновременно
доказанной working и sealed ready DB. Несколько записей одного события и запись,
не появившаяся за 72 часа, остаются в карантине с явным способом разбора.

## Конфигурация

Канонический шаблон:
`docs/m1_calls_handoff_20260801/config.m1.example.json`.
При создании локального JSON каждый `<HOME>` заменяется фактическим `$HOME`, а
пути исходного Mac и M1 остаются разными явными параметрами. Слепая замена строк
в SQLite запрещена; используется только relocation-механизм.

Config и env — обычные файлы владельца `0600`, не symlink и без extended ACL.
Runtime находится под `$HOME/.mango_local`, секреты — под
`$HOME/.mango_secrets`; Git, iCloud и Яндекс.Диск для SQLite, аудио,
Codex-профиля и служебных журналов запрещены. Любой extended ACL на owner-only
cutover/watchdog evidence даёт отказ; код не снимает ACL автоматически.
Запущенный wrapper связывает дочерний процесс с SHA-256 проверенного поколения
config, поэтому замена файла между проверкой и фактическим caller даёт отказ.

Ключевые значения пилота:

- повторный просмотр Mango — 72 часа;
- перекрытие окон — 30 минут;
- ожидание аудио — до 72 часов, сигнал после 60 минут;
- каждая тяжёлая команда имеет собственный тайм-аут 4 часа;
- пустой worker завершается после одного idle-цикла;
- свободный диск — не менее 40 GiB;
- точный `expected_code_sha`, локальный `host_id`, явный
  `expected_previous_host_id`, cutover manifest v2 и путь к shutdown snapshot
  обязательны.

## Доказательство выключения старого Mac

Этот раздел относится только к будущему отдельно разрешённому cutover. Phase A
не создаёт реальное доказательство и не устанавливает службу.

`previous_host_shutdown_snapshot.json` — обычный файл владельца `0600`, не
symlink. Его SHA-256 должен дословно совпасть с
`cutover_manifest.json.previous_host_snapshot_sha256`. Snapshot связан с теми
же `previous_host_id`, `source_cursor_sha256`, `previous_host_disabled_at` и
`previous_host_checked_at` и содержит:

- полный scan всех 11 Calls launchd labels и пустой `active_calls_labels`;
- полный scan plist и пустой `active_calls_plists`;
- полный process scan версии `mango_calls_runtime_matchers_v1`, пустые PID и
  команды;
- полный cron scan и пустой `active_calls_cron_entries`;
- scan блокировок `process_a`, `capture`, `pipeline`, `process_b` и пустой
  `held_lock_names`.

Пустой JSON, чужой host, другой cursor, неполный scan, оставшийся plist/cron/PID
или изменённый SHA дают отказ. При первом закреплении cursor snapshot должен
быть свежим; далее lineage остаётся стабильным, но SHA, host, cursor и пустые
активные списки проверяются при каждом старте. Любая установка, содержащая
Process A, capture или pipeline, проходит эту проверку до обращения к launchd.
Render-only и отдельный Process B её не требуют.

Внешний watchdog — отдельный read-only наблюдатель вне обоих Mac. Локальный код
только валидирует его owner-only observation, привязанный к M1 host, старому
host, code SHA, cutover SHA и shutdown snapshot SHA. Наличие валидатора не
означает, что наблюдатель уже развёрнут: до этого runtime/cutover остаются STOP.

## Команды без установки

До отдельного допуска допустимы только чтение, dry-run и синтетика:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_mango_calls_stage10_verdict.py \
  tests/test_mango_calls_two_processes.py \
  tests/test_relocate_mango_calls_pipeline.py

python3 scripts/install_mango_calls_two_processes_service.py \
  --config "$HOME/.mango_local/mango_calls_two_processes/config.json" \
  --env-file "$HOME/.mango_secrets/mango_calls_m1_worker.env" \
  --fast-service \
  --out-dir "$HOME/.mango_local/mango_calls_plist_review"
```

Вторая команда только отрисовывает plist. Флаг `--install` без отдельного
разрешения запрещён. Строгий M1 config требует явный `--fast-service`,
`--process-a-only` или `--process-b-only`: старые команды wrapper без суффикса
`-worker` для такого config отклоняются и не являются fallback-топологией.

После разрешения на конкретный этап CLI имеет отдельные команды:

```text
run_mango_calls_pipeline.py --config CONFIG capture
run_mango_calls_pipeline.py --config CONFIG pipeline
run_mango_calls_pipeline.py --config CONFIG watchdog
```

`capture` не пишет working SQLite и не запускает модели. `pipeline` фиксирует
префикс capture manifest и не видит записи, добавленные после старта. Второй
тяжёлый процесс немедленно получает занятый `flock`, а timeout завершает всю
группу процессов, включая дочерние процессы стадии.

## Расписание fast-service

Отрисованный комплект содержит:

- capture: `:00`, `:15`, `:30`, `:45`;
- pipeline: `:07`, `:37`;
- локальный watchdog: `:12`, `:27`, `:42`, `:57`;
- попытки закрытия: 06:00, 07:00, 08:00 МСК;
- обезличенный сигнал: 08:30 МСК;
- обязательный локальный статус: 08:50 МСК;
- Process B: demand-only, без собственного расписания.

Plist Process B нужен только для явного ручного повтора/восстановления;
по расписанию его никто не запускает. Обычный caller — `pipeline`, который вызывает
Process B напрямую после запечатанного ready-поколения.

После каждого успешного pipeline создаётся только локальный short-lived
safe-plan текущего дня и повторно проверяются незакрытые сутки последних 72
часов. Google и Яндекс.Диск из plist не вызываются.

## Локальная публикационная подготовка

`scripts/run_mango_calls_publication_coordinator.py` создаёт только файлы под
`$HOME/.mango_local/mango_calls_publication`:

- `current-plan` — manager-проекция без полного диалога и путей БД;
- `daily-close` — локальный финальный пакет только при `closure_ok=true`;
- `daily-alert` — агрегат без телефона, ФИО, текста и путей;
- `daily-status` — всегда `final`, `incomplete` или `failed`, даже при ошибке
  ready manifest/export.

Safe-plan действует не более 60 минут и связан с exact ready SHA и Stage 10.
Будущий ручной Google execute дополнительно требует подтверждение и SHA-256
ровно этого safe-plan. Эти параметры не входят в Phase A env и автоматически
не вызываются.

Неполный суточный пакет имеет префикс
`НЕПОЛНЫЙ, НЕ ИСПОЛЬЗОВАТЬ КАК ИТОГ`. Он строится из консистентного SQLite
backup, а не копированием живых WAL-файлов. Позднее зелёное закрытие создаёт
отдельный final/supplement и не переписывает прежнее поколение.

## Stage 10 и контроль

Для каждого московского дня проверяется:

```text
mango_unique = ready_unique + quarantine_unique + pending_unique
             + unexplained_missing
```

`consistency_ok` допускает свежий pending, но требует полное перечисление,
нулевые дубли/пересечения/необъяснённые пропуски и объяснённый карантин.
`closure_ok` дополнительно требует полный день, `pending=0`, dual ASR (или
полное утверждённое исключение), Resolve и Analyze. Пустой день требует два
независимых полных нулевых перечисления.

## Переходы между этапами

- Phase A → B: зелёные синтетические тесты, audit pack и независимый аудит.
- B → controlled-10: отдельное разрешение на зависимости, probe и синтетические
  model-пробы; никакого launchd.
- controlled-10 → реальный день: отдельное разрешение на 1, затем 10 звонков и
  внешнюю обработку текста.
- постоянная служба: только после трёх дней РОПа и отдельного разрешения на
  launchd/cutover.

Локальный manifest не является межмашинным замком. До cutover требуется свежее
доказательство остановки старого Process A и внешний read-only наблюдатель.
Без него статус cutover остаётся STOP.
