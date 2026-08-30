> DONE 2026-08-30 22:05 | ветка codex/customer-timeline-single-writer-20260830 | codex

> TAKE 2026-08-30 19:34 | ветка codex/customer-timeline-single-writer-20260830 | codex

Ветка: codex/customer-timeline-single-writer-20260830
Зоны: src/mango_mvp/customer_timeline/, scripts/publish_snapshot/, scripts/run_customer_timeline_*.py, scripts/build_customer_timeline_nightly_dv2_sources.py, tests/test_customer_timeline_*.py, tests/test_publish_snapshot_tooling.py, docs/, tasks/, .claude/audit_inputs/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_publish_snapshot_tooling.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_codex_task.py tests/test_customer_timeline_read_api.py tests/test_customer_timeline_manager_dossier.py tests/test_draft_loop.py
Семантический-аудит: да
Feature-ID: customer_timeline.single_writer_release
Problem-ID: problem.customer_timeline.dual_writer_release_gates
Исход: attempt_complete
Изменение: extend
Ключевые-символы: build_compact_reader,bot_visibility_counts,flip,CustomerTimelineSQLiteStore,run_customer_timeline_nightly_service,ensure_nightly_config
Ключевые-слова: Customer Timeline single writer,compact size gate,effective bot visibility,expected snapshot SHA,M1 M4 cutover,nightly overlap

# ТЗ: один Customer Timeline writer и три обязательных release-gate

Дата: 2026-08-30.
## Цель

Свести две расходящиеся nightly-линии Customer Timeline в одну каноническую
линию на M4, не потеряв более свежий хвост M4 и не создавая третьего
конвейера. Закрыть три найденных ложнозелёных release-проверки до любого
production flip:

1. размер compact-reader;
2. эффективная видимость памяти после канонического reader;
3. соответствие переключаемого SQLite ожидаемому manifest/SHA.

## Исходная правда

- Текущий `main` M4 содержит старую nightly-службу и локальную staging-базу.
- Переданный M1-кандидат: `1b3709f693239cedb66219793623f62202c11498`.
- Канонический донор — целиком дерево именно этого SHA. Коммиты, в которых
  отдельные символы впервые появились на других ветках, не являются
  дополнительными донорами и не должны интегрироваться отдельно.
- Frozen seed пакета:
  `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/M1_to_M4_customer_timeline_complete_handoff_20260830T080449Z`.
- M1-кандидат является продолжением `5e47b894` на 38 коммитов; текущий `main`
  дополнительно содержит только более новые независимые изменения, которые
  нельзя терять.
- SQLite вручную не объединять. M4-only хвост возвращать штатными адаптерами с
  безопасным overlap и стабильными dedupe-ключами.

## Жёсткие границы

- Клиентские отправки, CRM/AMO/Tallanto write, ASR и production flip: `0`.
- ПДн и секреты не коммитить и не выносить в audit/Foton.
- Существующую M4 staging, старый snapshot, rollback-пакеты и пакет M1 не
  удалять и не изменять.
- До доказанного единственного writer не запускать новый write-цикл.
- Нельзя ослаблять P0, identity, brand, family, Mail/Calls/Wappi source policy.
- Любой конфликт при интеграции общего live `draft_loop` проходит отдельный
  смысловой аудит; нельзя принимать ветку только по зелёным тестам.

## A. Принять одного владельца кода

1. Выполнить обязательный inventory всех веток/worktree и Graphify с raw-source
   перепроверкой.
2. Интегрировать точный M1-кандидат в рабочую ветку без повторной реализации
   nightly, resolver, compact builder или manager dossier.
   Сначала принять это дерево, и только затем менять три release-gate в уже
   существующих донорских `build_snapshot`, `reader_smoke` и `flip`.
3. Сохранить независимые более новые решения текущего `main`, включая `D-143`.
4. Проверить diff общего живого пути `src/mango_mvp/integrations/draft_loop.py`
   отдельным post-merge semantic review и точечными тестами до любого runtime
   действия. Предполетный пакет проверяет направление интеграции и release-gate;
   он не объявляется проверкой всех 38 входящих коммитов.
5. Сразу после merge повторить inventory по результирующему дереву, чтобы
   владельцами трёх gate стали интегрированные файлы текущей ветки, а не
   discovery-ссылки на донорские refs.

## B. Release-gate 1: размер compact

После интеграции M1-кандидата заменить его захардкоженный потолок, практически
равный текущему файлу, на явный бюджет релиза из publish-конфига. Сборка обязана:

- fail-loud при отсутствующем/некорректном бюджете;
- записывать фактический размер, бюджет и запас в manifest;
- отклонять искусственно раздутый compact;
- принимать текущий доказанный compact и разумный инкремент без ручной правки
  кода.

Единый разбор и валидация бюджета живут в `common.py`; значение хранится в
publish-config. Дублировать парсинг в `build_snapshot.py` запрещено.

## C. Release-gate 2: эффективная видимость

`reader_smoke` и `build_snapshot` обязаны проверять не только
`allowed_for_bot`, а итог после канонического reader:

- effective count и blocked count записываются в manifest;
- политика/ожидаемый диапазон задаются конфигом или принятым предыдущим
  manifest, а не молчаливой константой;
- эффективные сырые Mail/Calls/Wappi и ambiguous/unmatched равны нулю;
- пустая либо неожиданно расширенная память делает релиз красным;
- отрицательный тест ломает source-policy и доказывает, что gate краснеет.

## D. Release-gate 3: expected manifest/SHA при flip

Исполняемый `flip` принимает обязательный ожидаемый build manifest, сверяет:

- SHA snapshot-файла;
- размер и `user_version`;
- успешные integrity/FK/domain/reader gates;
- принадлежность manifest и SQLite одному каталогу сборки.

Без manifest, при несовпадающем SHA или ложном gate `--execute` запрещён.
Dry-run остаётся доступным для диагностики.

CLI, общая загрузка конфигурации и проверка manifest могут быть изменены в
`common.py` и publish-config; формулировка «три gate» не ограничивает исправление
тремя Python-файлами.

Если ошибка обнаружена уже после атомарной подмены, `flip` обязан автоматически
вернуть в prod проверенный локальный backup, сверить его SHA/quick_check и
оставить readers остановленными. Простого сообщения о пути к backup недостаточно.

## E. Single-writer и сохранение хвоста

1. Зафиксировать фактическую старую M4-службу: launchd, PID, HEAD, config, DB
   SHA, последний report и cursors. Старую DB только заморозить как rollback.
2. Восстановить M1 full seed в новый локальный owner-only runtime M4, не в
   Яндекс.Диск и не поверх старой DB.
3. Доказать M1 writer stop/seed freeze. Создать `WRITER_OWNERSHIP.json` только
   из проверенных фактов; receipt должен проверяться перед каждым mutating run.
4. Пересобрать один host-local nightly config существующим builder.
5. Вернуть M4-only хвост штатными AMO/Tallanto/Wappi/Calls/Mail адаптерами с
   overlap. Для каждого источника показать баланс и монотонность cursors.
6. Первый цикл догоняет дельту; немедленный второй цикл почти пустой, не создаёт
   физических дублей и не меняет доказанные связи без нового входа.
7. Только после двух циклов оставить загруженной ровно одну M4-службу на одном
   pinned SHA/runtime home. Старую plist/DB не удалять до отдельного cleanup.

## Приёмка

### Формальная и data

- Точечные тесты ТЗ и полный `pytest` без новых падений.
- Frozen seed и итоговая staging: `quick_check=ok`, FK=0, domain=PASS,
  sidecars=0 после checkpoint.
- `received = linked + quarantine + proven_junk + error` по изменившимся
  источникам; cursor не откатился и не ушёл в будущее.
- Повтор: новых бизнес-строк и физических дублей 0; допустимые timestamp/run-id
  изменения перечислены отдельно.

### Release

- Все три негативных теста реально краснеют при искусственной поломке.
- Отрицательный тест сбоя после `os.replace()` доказывает автоматическое
  восстановление прежнего prod-файла по ожидаемому SHA.
- Новый compact воспроизводим на одном cutoff.
- Reader canary `1 -> 10 -> 100`: raw запрещённых источников 0, false READY 0.
- Production pointer и живой бот не менялись.

### Смысл и бизнес

- На независимой обезличенной выборке менеджер видит доказанный контекст либо
  честное `нужна проверка`; неподтверждённое действие не становится READY.
- Отдельные verdict: formal, data, semantic, business, runtime.

## СТОП

- Не доказано, что M1 writer остановлен и frozen seed не изменился.
- Нельзя получить верифицированный rollback старой M4 staging.
- Интеграция меняет клиентский/live путь без semantic PASS.
- Любой из трёх release-gate остаётся ложнозелёным.
- Первый цикл теряет строки, откатывает cursor или не замыкает баланс.
- Любая операция потребовала бы отправки клиенту или записи в AMO/Tallanto.

## Результаты

- один канонический код nightly и один host-local config;
- ownership receipt и rollback-инструкция;
- три release-gate с тестами;
- отчёты двух циклов и compact canary;
- один audit pack;
- один коммит и push рабочей ветки;
- явный статус: `problem_closed` только если две линии фактически сведены и
  все три gate проходят; иначе `attempt_complete` с одним доказанным blocker.

## Бритва

Переиспользовать M1-кандидат и существующие publish/nightly helpers. Новый
builder, второй resolver, новый ingestion-путь или ручная миграция SQLite
запрещены. Нового нетестового кода для трёх gate — не более 150 строк без
отдельного пересмотра ТЗ.

Пересмотр 2026-08-30 после независимого breaker-review: потолок повышен до
170 строк. Причина — обязательный автоматический возврат проверенного backup
после сбоя, уже случившегося за границей `os.replace`; сокращать его ради
формального лимита небезопасно. Фактический дифф трёх gate — 166 строк,
нового ingestion/resolver/builder не добавлено.

Предполетные незакоммиченные файлы до merge ограничены этим ТЗ, строками
реестра worktree и локальными обезличенными входами Claude в
`.claude/audit_inputs/`; это доказательства процесса, не второй runtime-код.
