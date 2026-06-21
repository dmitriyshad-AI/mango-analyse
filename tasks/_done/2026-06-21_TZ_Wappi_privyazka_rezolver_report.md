# TZ Wappi Resolver Report

Дата: 2026-06-21

Ветка: `codex/wappi-history`

## Что сделано

- Перенёс `AmoAutoResolver` из `scripts/run_amo_wappi_draft_loop.py` в общий модуль `src/mango_mvp/integrations/amo_wappi_auto_resolver.py`.
- Подключил AMO auto-resolver к Wappi history ingest только для чатов без статичной пары.
- TG резолвится только по числовому Wappi `chat_id` -> точный `Telegram ID` в AMO; телефон для TG не используется.
- MAX резолвится только по телефону из dialog/participants и только при наличии shared-phone stoplist; без стоп-листа fail-closed.
- Цепочка привязки: Wappi chat -> AMO contact -> ровно 1 активная deal -> бренд deal == бренд профиля -> существующий `customer_id` в timeline. Новый клиент не создаётся.
- Добавлен fail-closed для `brand_unknown`, `brand_mismatch` и смешанной строки организации `Фотон + УНПК`.
- Добавлен guard: существующий Wappi `source_id` нельзя молча переклеить на другого `customer_id`; конфликт уходит в pending.
- `allowed_for_bot=0` сохранён для всех Wappi chunks/events.

## Тестовая заливка

Тестовая копия:

`product_data/customer_timeline/canonical_readonly_wappi_resolver_final2_testcopy_20260621T120450Z/customer_timeline.sqlite`

Отчёты:

- `product_data/customer_timeline/canonical_readonly_wappi_resolver_final2_testcopy_20260621T120450Z/wappi_resolver_apply_report.json`
- `product_data/customer_timeline/canonical_readonly_wappi_resolver_final2_testcopy_20260621T120450Z/wappi_resolver_apply_repeat_report.json`

Лимиты: 50 чатов на профиль, 50 сообщений на чат, общий потолок 1000 сообщений, `request-limit=350`, пауза 0.15 сек.

Итог apply: 842 записи, 55 привязано через AMO auto-resolver, 787 pending.

| profile_id | бренд | канал | записей | привязано | pending | покрытие |
|---|---:|---:|---:|---:|---:|---|
| `152b441d-81a2` | unpk | max | 199 | 0 | 199 | 9/50 MAX с телефоном вне стоп-листа |
| `18b255b8-7a67` | unpk | telegram | 250 | 49 | 201 | 20/20 TG numeric |
| `2952990f-9e4c` | foton | max | 143 | 0 | 143 | 16/50 MAX с телефоном вне стоп-листа |
| `ec2eed50-b55f` | foton | telegram | 250 | 6 | 244 | 26/26 TG numeric |

Pending reasons:

- `max_phone_missing`: 279
- `closed_lead`: 267
- `amo_auto_has_no_customer_in_timeline`: 116
- `telegram_id_no_contact`: 48
- `no_active_lead`: 49
- `brand_unknown`: 24
- `multi_active_lead`: 3
- `brand_mismatch`: 1

Важно: предыдущий Wappi batch был 854 записей, текущий live-read при тех же лимитах вернул 842 импортируемые записи. Это изменение источника Wappi во времени, не изменение боевой памяти.

## Идемпотентность

- Повторный apply: Wappi events/chunks/identity_links не выросли.
- Дубликаты Wappi `timeline_events` по `(source_system, source_id)`: 0.
- Дубликаты Wappi `bot_context_chunks` по `(source_system, source_ref)`: 0.
- Переклейка существующего `source_id` на другого клиента: 0 (`blocked_customer_relink_conflicts=0`).
- В repeat появился 1 новый pending-conflict из live-окна Wappi/AMO; это не дубль события и не переклейка. Для боевого шага лучше фиксировать snapshot входа или курсор before/after.

## NEG

- Wappi: только `DefaultDenyTransport`, GET, `mark_all=false`, лимит страницы <=100.
- AMO: только `AmoMcpClient.amo_api_get`, MCP GET-only; write-методов в новом модуле нет.
- AMO/Tallanto/CRM запись: 0.
- Сообщения клиентам: 0.
- Боевая customer_timeline DB не тронута; запись только в test-copy.
- `allowed_for_bot=1` у Wappi chunks: 0.
- Греп по Wappi rows на email/full phone: 0.
- MAX без стоп-листа или с общим телефоном не привязывается.
- TG по телефону не привязывается: покрыто тестом.
- Смешанный бренд в организации fail-closed: покрыто тестом.

## Проверки

- Preflight: OK.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_import_to_timeline.py tests/test_amo_wappi_auto_resolver.py tests/test_amo_wappi_transport.py tests/test_run_amo_wappi_draft_loop.py`: 24 passed.
- Полный pytest: 3495 passed, 5 skipped, 1 warning.
- SQLite `PRAGMA quick_check`: ok.
- Независимый semantic review: `PASS_WITH_NOTES`; блокеров нет, не production-ready.

## Риски

- 55 связанных сообщений соответствуют 2 Telegram-чатам; перед боевым apply нужен ручной spot-check этих сессий без вывода ПДн.
- Часть привязок может опираться на contact bridge при отсутствии lead/opportunity bridge в timeline; это допустимо для test-copy, но требует ручного контроля перед боем.
- Live Wappi/AMO не является неизменяемым snapshot; repeat может увидеть новые pending сообщения. Для боевого шага нужен фиксированный вход или курсор.

## Артефакты

- Audit pack: `audits/_inbox/wappi_resolver_20260621151600/`
- Raw/test SQLite и JSON-отчёты находятся в ignored `product_data/customer_timeline/canonical_readonly_wappi_resolver_final2_testcopy_20260621T120450Z/`.

В `main` не вливал. К боевой памяти не применял.
