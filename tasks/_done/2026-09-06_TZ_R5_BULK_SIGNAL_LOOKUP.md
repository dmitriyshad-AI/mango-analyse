> DONE 2026-09-06 14:33 | ветка codex/customer-timeline-source-gates-20260902 | codex

> TAKE 2026-09-06 13:41 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: src/mango_mvp/customer_timeline/store.py, tests/test_customer_timeline_store.py, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_store.py tests/test_customer_timeline_contracts.py tests/test_customer_timeline_ingestion.py tests/test_wappi_history_import_to_timeline.py tests/test_wappi_history_checkpoint.py tests/test_customer_timeline_nightly_service.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.bulk_signal_lookup
Problem-ID: problem.customer_timeline.m4_first_cycle_source_failures
Исход: attempt_complete
Изменение: fix
Ключевые-символы: _retire_signal_dependencies,bulk_write,upsert_signal
Ключевые-слова: signal source event dependency,bulk dependency lookup,quarantine owner change

# R5: убрать повторный полный поиск зависимостей при записи сообщений
## Доказательство

HEAD605161027, r5 уже законно активирован. AMO complete; Wappi остановлен по
просьбе владельца на store._retire_signal_dependencies. После остановки
577271 событий, quick/FK/WAL чистые. Новый readonly SELECT по заведомо
отсутствующей ссылке занял3.571s при118397 derived_signals. EXPLAIN использует
только tenant-prefix и json_each каждой записи; ix_signals_multi_source
не используется. Это подтверждает повторный дорогой поиск, не доказывает
все53минуты прошлой загрузки. Авторизовано продолжение текущего r5 без seed.

Существующий владелец выбран inventory: store.py::_retire_signal_dependencies,
decision=extend. Схожий принцип массовой работы уже есть у bulk FTS keys.

## Варианты

1. Один частичный индекс по multi-source: недостаточен, пропускает singleton.
   Индекс первого элемента плюс отдельные хвосты усложняет legacy-эквивалентность.
2. Постоянная таблица ссылок с миграциями/триггерами: больше изменений и новый
   постоянный контракт, для текущей задержки не требуется.
3. Временная индексированная карта ссылок на время bulk_write плюс прежний
   финальный EXISTS: выбран минимальный вариант, без новой постоянной схемы.

## Реализация

- В существующем Store лениво создать TEMP карту tenant/event/signal через
  тот же json_each и CAST(value AS TEXT). Карта только отбирает кандидатов;
  прежний EXISTS, owner/status и (event_id IS NULL OR event_id!=?) остаются
  окончательной проверкой. Прямые/вторичные ссылки не считаются дважды.
- Карта строится один раз на tenant за outer bulk, готовность хранится как
  set tenant. После фактической записи нового/изменённого derived_signal в
  _upsert_record (после duplicate-return) добавлять ссылки из нового payload_json
  через INSERT OR IGNORE/json_each в уже готовую карту этого tenant. Это
  включает repair физических колонок/JSON. Дубликат не пишет карту. Старые
  отозванные ссылки остаются кандидатами: прежний EXISTS отсечёт их; новые
  добавляются без повторного полного сканирования. Owner/status в карте нет.
- Карта заполняется лениво по tenant вызова; поддерживаются все tenant и
  корректные формы, ранее читаемые json_each. Ошибки JSON не замалчивать;
  не отбрасывать single/stale/resolved ради скорости. Owner=NULL и сохранение
  действительного владельца работают как прежде.
- Вне bulk сохранить прежний путь. Перед новым outer bulk и после выхода,
  включая исключение, готовность карты сбрасывается. Nested bulk не пересоздаёт
  карту; rollback/retry не используют ложную готовность. INSERT карты и записи
  выполняются внутри той же транзакции. Временная таблица не переносится между
  соединениями. После сброса готовности содержимое tenant перед сборкой очищать.
- Флаги хранения бота, strong-порог, карантин, source policy, отбор Wappi,
  сеть, транзакционная граница событий/FTS и курсор не менять.
- До50 добавленных обычных строк; тесты/защитные регрессии не урезать.
  Один внутренний helper/TEMP table допускаются этим ТЗ. Нет внешних флагов,
  зависимостей, служб, ручных индексов в r5 или нового формата SQLite.

## Приёмка

1. Одинаковый итог baseline/optimized на синтетике1/10/100: прямой/вторичный
   источник, singleton/multi/пусто, active/resolved/stale, разные tenant,
   owner change -> pending, сохранение действительного владельца. Итог
   включает счётчики аудита: прямой сигнал с тем же event_id в массиве
   учитывается один раз, не только получает правильный конечный JSON.
2. Добавление/изменение сигналов внутри bulk не теряет новых ссылок и не
   считает устаревшего кандидата актуальным.
   Отзыв не убирает другие источники; вторичный источник с event_id=NULL
   тоже отзывается. Физический repair не обходит пополнение карты.
3. Rollback, повтор и nested: состояния событий/сигналов/чанков/FTS совпадают
   с прежними; cached flag после outer выхода сброшен. Повтор без новых
   зависимостей не создаёт дубликаты и не сканирует весь JSON каждый раз.
   При N owner-change и M новых/изменённых signal в одном bulk на tenant
   разрешён один первоначальный полный проход; затем только point-add ссылок.
4. Проверка EXPLAIN/счётчика операций использует фактический runtime SQL,
   не искусственный запрос с удобным для индекса предикатом. Не требовать
   хрупкого абсолютного времени CI. Два старых EXPLAIN с вручную добавленным
   json_array_length>1 заменить трассой реального SQL, сохранив поведенческие
   assertions. Для real замера выполнять захваченный SELECT на соединении
   :memory: с ATTACH source mode=ro&immutable=1 после проверки пустого WAL;
   TEMP-карта только в памяти, baseline/optimized с одинаковыми входами.
   bulk_write на readonly Store не вызывать. Полный отзыв и транзакции
   проверяются на синтетической временной БД обычным API Store. Полная
   14GB-копия ради одного SELECT не нужна, main DB в замере не пишется.
   In-memory замер не выдавать за замер end-to-end реального writer: его
   temp_store/IO могут отличаться. Не вводить новое PRAGMA в рабочую базу.
5. Независимые architect/breaker/business/cleaner роли, code+semantic отдельно.
   Claude получает только код, фикстуры и обезличенные доказательства.
6. После кода коммит и стандартная code-release квитанция605161027->NEW;
   исходные ownership/activation не перепечатывать. Только затем источник
   Wappi из сохранённого состояния, доказательный итог и повтор. При новом
   измеренном bottleneck не запускать цикл снова вслепую.

## Границы завершения

Этот фикс не закрывает родительскую цель сам по себе. Два строгих полных
цикла, compact/readers и приёмка обоих Mac плюс одно расписание остаются.
Уборка отдельных копий выполняется по отдельному журналу и не смешивается
с коммитом кода. Не трогать prod/live/CRM/ASR/клиентские отправки.

## СТОП

Занятый writer/неподтверждённый ownership, чужие изменения, потеря ссылок
или расширение видимости, расхождение baseline/optimized останавливают
применение. Не менять полы и не выдавать partial за успешный цикл.

## Уточнение после двух независимых pre-review

Повторный review нужен не для получения желаемой оценки того же плана:
реализация изменена с invalidate/rebuild на инкрементальное пополнение карты,
добавлена приёмка смешанного потока N event / M signal. Приняты замечания о
per-tenant готовности, неизменном EXISTS, audit-count и реальном EXPLAIN.
Не подтверждена фраза аудитора про чередование signal/event именно в Wappi:
wappi_history_import.py возвращает TimelineNormalizedBatch без signals;
ingestion.py::ingest_stream держит bulk_write. Общий Store всё равно обязан
поддерживать смешанный поток других нормализаторов, поэтому его замечание
о возможных повторных перестроениях устранено в общей реализации.
SIGINT/изменение catch BaseException не входит в этот фикс. Старый частичный
индекс не удалять здесь: отдельный аудит необходимости, не часть ускорения.

## Результат кодовой попытки 06.09.2026

Outcome: attempt_complete
Evidence: audits/_inbox/customer_timeline_r5_signal_lookup_20260906/
Formal: 761 passed (6 профильных файлов + мораторий), 1 urllib3/LibreSSL warning.
Локальный архитектурный аудит: PASS после исправления TEMP total_changes;
baseline/optimized synthetic и реальный readonly SELECT совпали.
Store +42/-3, без постоянной схемы/новых флагов/зависимостей.

Это завершение кодовой попытки, НЕ всей задачи эксплуатации. Пункт 6 требует
post-commit review -> code-release -> реальный источник -> повтор. Этот
операционный хвост выполняется в продолжающейся родительской задаче
tasks/_running/2026-09-06_TZ_R5_SOURCE_PAGINATION_COMPLETION.md и активной цели,
на том же r5 без seed. Problem-ID не закрыт. Полных строгих циклов пока 0/2,
compact и расписание ещё не приняты. Не выдавать замер SELECT за Wappi throughput.
