> TAKE 2026-09-02 13:32 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: scripts/backfill_customer_timeline_next_steps_from_summary.py, scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/repair_mail_stage2_event_dates.py, scripts/retrofit_channel_brand_tags_in_timeline.py, scripts/run_customer_timeline_codex_task.py, scripts/run_customer_timeline_mail_chain.py, scripts/run_customer_timeline_mail_import.py, scripts/run_customer_timeline_nightly_incremental.py, scripts/run_customer_timeline_nightly_service.py, deploy/customer_timeline_daily_captures/, src/mango_mvp/customer_timeline/family_graph.py, src/mango_mvp/customer_timeline/nightly_incremental.py, src/mango_mvp/customer_timeline/mail_stage2_ingest.py, src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/customer_timeline/safety.py, src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/wappi_history_import.py, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_codex_task.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_contracts.py tests/test_customer_timeline_family_graph.py tests/test_customer_timeline_mail_stage2_ingest.py tests/test_wappi_history_checkpoint.py tests/test_wappi_history_import_to_timeline.py tests/test_retrofit_channel_brand_tags_in_timeline.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.stable_source_gates
Problem-ID: problem.customer_timeline.m4_first_cycle_source_failures
Изменение: fix
Ключевые-символы: load_incremental_jsonl_source,parse_mail_stage2_event_at,wappi_fetch_universe_fingerprint,ensure_nightly_config,managed_staging_writer_scope,guard_managed_customer_timeline_staging_write,activate_writer_ownership_after_success,validate_writer_ownership,run_nightly_service,resolve_chat,IncrementalSourceConfig,status_from_payload
Ключевые-слова: mail event timestamp proof,Wappi checkpoint fingerprint,immutable pair snapshot,single writer activation boundary,Tallanto subprocess scope,partial cycle recovery,Wappi owner conflict,mail proof manifest,raw sqlite writer bypass

Проверенный-донор: e4391b438c9a25fa544cf0d4adc680593d9897bb

# Customer Timeline: закрыть два сбоя первого цикла M4

## Примечание для preflight

Текущий HEAD — исходная ревизия до исправлений. Preflight оценивает, полон ли, безопасен ли и
реализуем ли план ниже. Нереализованные пункты 14–18 на исходном HEAD сами по себе не являются
`PASS_WITH_FIXES`: это и есть заданная работа. `PASS_WITH_FIXES` нужен только если для их корректной реализации в ТЗ
отсутствует обязательное решение или граница.

## Проблема

Первый цикл M4 остановлен двумя обязательными источниками:

1. Mail: producer декларирует `max_event_at` по дате письма, а proof-проверка
   сравнивает его с `updated_at`, используемым как курсор источника.
2. Wappi: host-builder заменил переданное M1 значение `messages_per_chat=50000`
   на `100`, сбросил совместимый checkpoint и запустил полный аудит. Во время
   чтения общий auto-pairs файл изменился живым контуром, поэтому provenance
   gate правильно заблокировал применение.

## Сделать

1. Переиспользовать `parse_mail_stage2_event_at` из полного проверенного SHA
   `e4391b438c9a25fa544cf0d4adc680593d9897bb`: экспортировать его из
   `mail_stage2_ingest.py`, импортировать в `nightly_incremental.py` и применять
   только для даты email-события, стабильного email-digest и доказательства
   `max_event_at`. Общий `normalized_timestamp` не менять: `updated_at` остаётся
   курсором и служебной свежестью источника.
2. Восстановить checkpoint-совместимое `messages_per_chat=50000` при
   `complete_message_history=true`.
3. Перед каждым nightly запуском атомарно снимать локальные копии manual/auto
   Wappi pair-файлов в owner-only staging и направлять importer только на них.
   Живые pair-файлы не менять. Переиспользовать существующие
   `load_pairs_file_snapshot` для стабильного чтения/проверки JSON и
   `atomic_publish_latest` для атомарной локальной публикации; не создавать
   второй парсер или второй Wappi-importer.
4. Валидатор nightly-конфига обязан остановить запуск до сети, если
   `messages_per_chat != 50000`, `complete_message_history != true` или pair-пути
   направлены не в staging-снимки.
5. Снимок каждой pair-карты принимать только после совпадения SHA исходного
   стабильного чтения и локальной копии; невалидный JSON или гонка дают STOP.
6. Добавить положительные, соседние и враждебные тесты: `event_at < updated_at`,
   `event_at == updated_at`, ложный `max_event_at`, несовместимый Wappi-конфиг и
   изменение живого auto-pairs после создания frozen-входа.
7. По находке независимого архитектора включить SHA обоих frozen pair-файлов в
   конфигурацию и fingerprint nightly-run. Импортёр сверяет эти SHA до сетевого
   чтения; изменение pair-набора запрещает resume старого запуска.
8. По находкам ломателя фильтровать Mail producer по техническому `updated_at`,
   сохраняя исходный `event_at`: поздно пришедшее старое письмо не теряется.
9. Pair-снимки сделать неизменяемыми и адресуемыми по содержимому. Новую
   pair-связь применять к уже сохранённой переписке по каталогу чатов без
   повторной загрузки полной истории; спорная смена владельца остаётся blocked.
10. Разделить стабильный fingerprint владения writer и полный fingerprint
    конкретного запуска: меняющиеся SHA Mail/pair запрещают resume старого run,
    но не ломают законный следующий ночной цикл после активации writer.
11. Закрыть найденные аудитом соседние пути второго writer: дневная Mail-цепочка
    только скачивает и обрабатывает архив, а запись в каноническую staging-БД
    выполняет единственный nightly. Предзапуск nightly сериализовать до сборки
    конфига; восстановление activation разрешать только по уже сохранённому
    зелёному отчёту и совпадающему снимку БД.
12. Каноническая staging-БД с ownership-квитанцией принимает write-mode через
    общий `Store` только из scope единого nightly. Прямые операционные CLI не
    могут стать вторым writer; тестовые и временные БД без ownership-квитанции
    сохраняют прежний контракт.
13. `pass_with_notes` не считается одним из двух зелёных ручных циклов и не
    обновляет last-success штатной обёртки. Квитанция владения может при этом
    зафиксировать уже состоявшийся безопасный commit: владение writer и
    приёмка качества являются разными состояниями.
14. На первом M4-цикле владение writer фиксируется после lock, проверки seed/stop-квитанции,
    пустого WAL и совместимости Wappi-checkpoint, но до первой записи. Право writer и зелёный
    статус цикла остаются разными фактами; partial не сжигает runtime-root.
15. Авторизованный nightly-scope пересекает границу subprocess только для штатного Tallanto-importer;
    тот же импортёр вне nightly-scope по-прежнему не может писать в managed staging.
16. Все пять найденных raw-SQLite путей перед apply/restore проходят общую пару managed+writable guards;
    read-only/dry-run контракты не меняются.
17. Запомненный `existing_wappi_chat_customer_conflict` не может быть затёрт последующей widget-резолюцией.
18. Обязательный Mail-источник всегда имеет существующий проверенный manifest и закреплённый SHA;
    ноль новых писем представлен пустым валидным manifest, а не отсутствием proof.

## Границы

- Никакой записи в production, AMO, Tallanto или Wappi; клиентских отправок нет.
- Текущий r4 и его failed-run не менять.
- После кода новый runtime восстанавливается из исходного frozen M1 seed.
- Proof, pagination и provenance gates не ослаблять.
- После нового коммита почтовые download/process-манифесты должны иметь одну
  runtime-идентичность. Использовать сохранённые локальные архивы и только
  штатный incremental read; не повторять полный исторический download.

## СТОП

- Потребовалось ослабить proof, pagination, identity или provenance gate.
- Потребовалась запись в production, AMO, Tallanto, Wappi или отправка клиенту.
- Новый запуск потребовал продолжить изменённый r4 без валидной activation-квитанции.
- M1 seed, stop-receipt или ownership-контракт не проходят повторную проверку.

## Приёмка

- mail `event_at < updated_at` проходит proof, событие получает `event_at`,
  а cursor продолжает использовать `updated_at`;
- соседний mail `event_at == updated_at` продолжает проходить без изменения
  прежней семантики;
- ложный `max_event_at` по-прежнему блокируется;
- Wappi fingerprint совпадает с переданным checkpoint;
- изменение живого auto-pairs во время импорта не меняет frozen-вход запуска;
- полный профильный pytest зелёный и независимый аудит не имеет P0/P1.
- первая activation-квитанция существует до первого write-Store; kill/partial не блокируют следующий запуск;
- дочерний Tallanto-importer пишет в managed staging только из проверенного nightly-scope;
- raw-SQLite apply/restore вне nightly-scope, затирание Wappi-конфликта и unpinned Mail-manifest закрыты регрессионными тестами.
