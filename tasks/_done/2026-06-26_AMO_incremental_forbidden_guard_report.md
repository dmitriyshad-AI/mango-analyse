# AMO incremental BOT_FORBIDDEN guard fix

Дата: 2026-06-26
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_amo_incremental_guard`
Ветка: `codex/amo-incremental-forbidden-guard`
Base: `main@95e6018`

## Статус

Выполнена только Часть 1 ТЗ: safety-фикс guard для AMO incremental.

Production Customer Timeline, AMO, Tallanto, CRM и live-бот не трогались. Боевое применение AMO incremental не запускалось.

## Что изменено

- В `BOT_FORBIDDEN_SOURCE_SYSTEMS` добавлены AMO incremental source systems:
  - `amo_events_created_at`
  - `amo_leads_updated_at`
  - `amo_contacts_updated_at`
  - `amocrm_event`
- `assert_bot_context_not_allowed_for_restricted_source()` больше не делает ранний `return` до проверки событий и чанков.
- Guard теперь всегда проверяет:
  - `source_record.payload`, если source system запрещён;
  - `event.source_system` + `event.record`;
  - `chunk.source_system` + `chunk.allowed_for_bot`.

## Почему это важно

Старый guard мог пропустить опасный случай: исходная запись называлась `amo_events_created_at`, а нормализованные event/chunk уже имели `source_system='amocrm_event'`. Из-за раннего выхода дочерние event/chunk не проверялись.

## Регресс-тест

Добавлен тест:

`test_amo_incremental_sources_reject_allowed_for_bot_true_in_children`

Он проверяет, что AMO event/chunk с `allowed_for_bot=True` теперь падает с ошибкой, даже если родительский source record называется `amo_events_created_at`.

## Проверки

Целевые тесты:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_customer_timeline_ingestion.py \
  tests/test_customer_timeline_amo_incremental.py \
  tests/test_customer_timeline_nightly_incremental.py

27 passed in 1.27s
```

Полный pytest:

```text
3659 passed, 5 skipped, 1 warning in 117.94s
```

`git diff --check`: чисто.

## Границы

- Production `customer_timeline.sqlite`: write `0`
- AMO/Tallanto/CRM write: `0`
- Messenger/client sends: `0`
- live-write scripts: не запускались
- Часть 2 ТЗ, production apply/swap: не запускались

## Следующий шаг

Регрейд Claude #1 по сырью. Боевое применение AMO incremental — только отдельным разрешением Дмитрия и только по свежему runbook с backup/SHA/apply-copy/checks.
