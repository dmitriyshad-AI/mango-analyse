> DONE 2026-08-04 00:18 | ветка codex/business-journey-integration-20260803 | codex

> TAKE 2026-08-03 23:27 | ветка codex/business-journey-integration-20260803 | codex

Ветка: codex/business-journey-integration-20260803
Зоны: tests/test_business_journey_traces_wave1.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_business_journey_traces_wave1.py tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py tests/test_amo_wappi_phase1.py tests/test_amo_wappi_transport.py tests/test_subscription_llm_draft_provider.py tests/test_bot_safe_runtime_context.py tests/test_dialogue_memory.py tests/test_p0_money_promise_output_floor.py tests/test_direct_p0_text_hygiene.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# Интеграция полезного из ветки сквозных бизнес-сценариев

## Исходная точка

- база: `main@1863b413ca341f215b2a93f405077becf35fb62d`;
- донор: `codex/business-journey-traces-wave1-20260803@70829e01884f1885c6ff671b7aeb64d29250c2d6`;
- донор добавляет 497 строк, в том числе новый `BoundaryTrace` и четыре сценария;
- прямой перенос заблокирован независимыми архитектурным, ломающим и
  бизнес-аудитами.

## Образ результата и бизнес-польза

Один безопасный синтетический прогон должен пройти фактический внутренний путь
`Wappi run_once -> штатный context builder -> SubscriptionLlmDraftProvider.build_draft ->
AiOfficeAmoNoteClient` и сформировать ровно одну настоящую по формату заметку
менеджеру. По её тексту без чтения кода видно:

- бот использовал память именно привязанного клиента и факт активного бренда;
- итог содержит полезный ответ, маркер `не отправлено`, бренд и маршрут;
- чужой бренд и внутренние идентификаторы отсутствуют;
- ни один клиентский send-метод не вызывался.

Второй сценарий проводит несловарный денежный спор по тому же пути: смысловой
кадр модели помечает P0, а менеджер получает `manager_only`-заметку без
модельного обещания возврата и без отправки клиенту. Проверяется результат, а
не только метаданные.

## Инвентаризация и выбранный вариант

1. Полный `cherry-pick` 497 строк: отклонён. Три сценария слабее существующих
   owner-тестов, реестр worktree дублируется, журнал перезаписывает evidence.
2. Новый общий tracing framework: отклонён. В проекте уже есть
   `dialogue_debug_trace`, `DraftLoopJournal` и semantic reading trace.
3. Выбрано: переписать донорский тестовый файл в два компактных сквозных
   сценария, переиспользовать существующие фикстуры и настоящие production
   adapters с локальным transport-stub. Новых production-механизмов нет.

## Что не переносить

- `BoundaryTrace`, `business_journey_boundary_trace_v1` и
  `MANGO_BUSINESS_TRACE_DIR`;
- отдельные identity/P0/Wappi проверки, уже сильнее покрытые в owner-модулях;
- синтетическую комбинацию `онлайн = 74 500 ₽`: в текущей KB это не действующий
  онлайн-факт;
- ручные названия fake-классов как будто это production adapters.

## Приёмка

1. Тест использует `scripts/run_amo_wappi_draft_loop.py::build_context_builder`,
   а не упрощённую лямбду.
2. AMO-часть использует `AiOfficeAmoNoteClient` и локальные transport/readback
   doubles; внешних HTTP-вызовов нет.
3. Позитивный сценарий проверяет фактический prompt и фактическое тело заметки:
   память своего клиента и факт Foton присутствуют, UNPK и внутренний customer ID
   отсутствуют, заметка непустая и помечена как неотправленная.
4. P0-сценарий проверяет финальные `manager_only`, безопасный текст заметки,
   отсутствие обещания вернуть деньги и `no_auto_send`; не закрепляет число
   модельных вызовов или конкретный regex-путь.
5. Все дополнительные модельные пути явно выключены или имеют tripwire:
   загрязнённое окружение не может тихо вызвать LLM.
6. Send-capable методы Wappi — ловушки; любая попытка отправки клиенту красит
   тест.
7. Отрицательные контроли доказывают: удаление customer memory из context,
   обход output/P0-route или подмена тела AMO-заметки делают тест красным.
8. Целевые, соседние owner-тесты и ADR003-мораторий зелёные.

## Запреты

- не менять `src/**`, runtime, базы, launchd и внешние системы;
- не запускать LLM, Wappi, AMO, Tallanto или клиентские отправки;
- не переносить ветку целиком и не создавать вторую трассировочную систему;
- не удалять донорскую ветку/worktree без отдельного решения владельца.

## СТОП

- для прохождения теста нужна правка production-кода;
- текущий `main` или donor SHA изменились;
- тест не может доказать фактический текст заметки и ноль клиентских отправок;
- независимый ломатель воспроизводит same-brand чужую память, чужой факт,
  неподтверждённые деньги или пропуск опасного P0.
