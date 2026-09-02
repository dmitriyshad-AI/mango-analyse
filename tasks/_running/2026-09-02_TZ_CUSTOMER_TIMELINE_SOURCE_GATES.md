> TAKE 2026-09-02 13:32 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/run_customer_timeline_codex_task.py, scripts/run_customer_timeline_mail_chain.py, scripts/run_customer_timeline_mail_import.py, scripts/run_customer_timeline_nightly_incremental.py, deploy/customer_timeline_daily_captures/, src/mango_mvp/customer_timeline/nightly_incremental.py, src/mango_mvp/customer_timeline/mail_stage2_ingest.py, src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/customer_timeline/safety.py, src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/wappi_history_import.py, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_codex_task.py tests/test_wappi_history_checkpoint.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.stable_source_gates
Problem-ID: problem.customer_timeline.m4_first_cycle_source_failures
Изменение: fix
Ключевые-символы: load_incremental_jsonl_source,parse_mail_stage2_event_at,wappi_fetch_universe_fingerprint,ensure_nightly_config
Ключевые-слова: mail event timestamp proof,Wappi checkpoint fingerprint,immutable pair snapshot

Проверенный-донор: e4391b438c9a25fa544cf0d4adc680593d9897bb

# Customer Timeline: закрыть два сбоя первого цикла M4

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
