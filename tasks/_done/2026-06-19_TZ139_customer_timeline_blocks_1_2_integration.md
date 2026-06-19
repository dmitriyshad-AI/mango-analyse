# TZ139 customer timeline blocks 1-2 integration

Дата: 2026-06-19

Ветка-источник: `codex/tz139-customer-timeline`
Интеграционная ветка: `codex/tz139-customer-timeline-integrate`
Итоговая вершина, отправленная в `origin/main`: `d15ace6`

## Что влито

- Work A: identity mapping + real-data safeguards.
- Work B0: family Mango phone links.
- Pre-B1 gate: инвариант каналовых чанков + single-writer lock.
- Work B1: Telegram timeline import.
- Work B2: Tallanto payments import.
- Work B3: WhatsApp/Max channel contract.
- Work C: derived signals lifecycle + deterministic signal derivation.

## Safety gates

- Каналовые чанки Telegram/WhatsApp/Max проверяются как `allowed_for_bot=False` и `requires_manager_review=True`.
- Bot retrieval/search, включая FTS и fallback, проверен на возврат только bot-safe chunks при `allowed_for_bot=True`.
- `CustomerTimelineSQLiteStore` теперь держит межпроцессный writer lock: один writable store на DB, read-only observers разрешены.
- `TimelineImportService.start_ingestion_run()` перенесён внутрь того же `bulk_write`, что и batch-write.
- B2: суммы Tallanto остаются в сигналах/manager-review слое, не как bot-safe chunks.

## Целевая БД

Путь:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_migration_20260618_002_copy/customer_timeline.sqlite`

Migration:

`20260618_002_derived_signal_status`

Writer policy:

один batch-writer через `CustomerTimelineSQLiteStore` + межпроцессный `customer_timeline.sqlite.writer.lock`; WAL остаётся для read-only наблюдателей, но не считается защитой от параллельных писателей.

DB quick_check: `ok`

Счётчики после сборки и C-apply:

- `customer_identities`: 16901
- `identity_links`: 59594
- `timeline_events`: 126209
- `bot_context_chunks`: 85189
- `customer_id_mappings`: 18164
- `derived_signals`: 414
- `ingestion_runs`: 1

Derived signals:

- `hot_lead_silent_7d`: 414
- `active`: 30
- `stale`: 384
- повторный `--apply` не увеличил число строк: второй прогон дал `updated=414`, новых дублей нет.

## Max read-only check

Wappi Max read-only проба:

- Max chat list/message endpoints доступны через `/maxapi/sync/chats/get` и `/maxapi/sync/messages/get`.
- Текст сообщения есть в `body`.
- Структурный телефон есть не во всех чатах: в sample 10 чатов Фотона `dialog/participant phone` был у 5, у УНПК у 1; поле `message.phone` было заполнено в большинстве проверенных сообщений.
- `crm_entities` есть, но это служебные поля `chat_id/crm_id/crm_type/manager_id/message_id`, не готовый `lead_id`.

Вывод: Max можно включать в timeline только как каналовый источник через Wappi/резолв пары. Если структурного телефона нет, авто-резолв остаётся `max_phone_missing`; брать телефон из текста переписки запрещено.

## Family phone sample

Приватная выборка для регрейда B1:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_migration_20260618_002_copy/family_phone_sample_for_b1_regrede_private.json`

В отчёт телефоны не включены.

## Tests

После Work A:

`3336 passed, 5 skipped`

После Work B0:

`3336 passed, 5 skipped`

После pre-B1 invariant:

- targeted: `4 passed`
- full: `3338 passed, 5 skipped`

После Work B1:

- targeted: `12 passed`
- full: `3340 passed, 5 skipped`

После Work B2:

- targeted: `22 passed`
- full: `3345 passed, 5 skipped`

После Work B3:

- targeted: `10 passed`
- full: `3346 passed, 5 skipped`

После Work C / финальная интеграционная вершина:

- targeted C: `12 passed`
- final full: `3362 passed, 5 skipped`

Примечание: предупреждение pytest одно и то же, `urllib3 NotOpenSSLWarning` из локального LibreSSL Python.

## Addendum: customer_profile contract for manager card

Добавлено в конце A:

- `customer_profile.snapshot_as_of` и `customer_profile.last_event_at` — детерминированная точка снимка от последнего события клиента, без `now()`.
- `customer_profile.customer_id_mappings` — read-only проекция `old_customer_id -> new_customer_id`.
- `timeline.items`, `signals`, `bot_context.items` явно несут `allowed_for_bot` и `requires_manager_review`.

Тесты:

- targeted: `tests/test_customer_timeline_read_api.py::test_read_api_profile_projects_safe_customer_timeline` → `1 passed`
- full: `3362 passed, 5 skipped`
