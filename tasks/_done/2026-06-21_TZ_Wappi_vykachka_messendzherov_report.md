# TZ Wappi History Ingest Report

Дата: 2026-06-21

Ветка: `codex/wappi-history`

## Что сделано

- Реализован read-only Wappi ingest для Telegram/Max по 4 профилям из `~/.mango_secrets/amo_wappi_phase1.json`.
- Все Wappi-вызовы идут через `DefaultDenyTransport`; `transport=None` для импортёра запрещён.
- Ужесточён Wappi safe transport: только GET, `limit<=100`, для сообщений `mark_all` должен быть явно false.
- Добавлены `wappi_telegram` и `wappi_max` в запрет bot-safe: `allowed_for_bot=True` отклоняется.
- Писатель использует существующий `TimelineImportService`; второй путь записи в память не добавлен.
- Резолв: `profile_id+chat_id -> draft_loop_pairs -> amo_lead_id/amo_contact_id -> existing customer_id`. Телефон не используется.

## Тестовая заливка

Тест-копия БД:

`product_data/customer_timeline/canonical_readonly_wappi_history_testcopy_20260621T104758Z/customer_timeline.sqlite`

Лимиты:

- 50 чатов на профиль;
- 50 сообщений на чат;
- общий потолок 1000 сообщений;
- fair-cap до 250 записей на профиль;
- 350 запросов максимум;
- пауза 0.15 с.

Счётчики:

| profile_id | бренд | канал | сообщений | привязано | pending |
|---|---:|---:|---:|---:|---:|
| `152b441d-81a2` | unpk | max | 212 | 0 | 212 |
| `18b255b8-7a67` | unpk | telegram | 250 | 0 | 250 |
| `2952990f-9e4c` | foton | max | 142 | 0 | 142 |
| `ec2eed50-b55f` | foton | telegram | 250 | 0 | 250 |

Итого: 854 записей, все `pending_attribution`. Это ожидаемый fail-closed: статические пары Wappi не дали единственного `customer_id` в тестовой памяти.

## Проверки

- Повторный apply: `duplicate=854`; число Wappi pending-конфликтов осталось 854.
- `PRAGMA quick_check = ok`.
- `allowed_for_bot=1` по Wappi chunks: 0.
- Wappi events/chunks: 0, потому что нет уверенной привязки к customer_id.
- Примеры в отчёте обезличены: текст сообщений скрыт, email/телефоны не найдены.
- Записей в AMO/Tallanto/CRM нет; stable_runtime не трогался; сообщения не отправлялись.

## Тесты

- `tests/test_amo_wappi_transport.py tests/test_wappi_history_import_to_timeline.py`: 9 passed.
- Wappi/draft-loop/customer_timeline/Telegram соседние тесты: 50 passed.
- После fair batching: 33 passed.
- Полный pytest: `3489 passed, 5 skipped, 1 warning`.

## Артефакты

- Apply report: `product_data/customer_timeline/canonical_readonly_wappi_history_testcopy_20260621T104758Z/wappi_apply_report.json` (ignored).
- Repeat report: `product_data/customer_timeline/canonical_readonly_wappi_history_testcopy_20260621T104758Z/wappi_apply_repeat_report.json` (ignored).
- Audit pack: `audits/_inbox/wappi_history_20260621135454/` (ignored).

## Что безопасно коммитить

- Код импортёра Wappi history.
- Тесты.
- Изменение `.gitignore` не потребовалось: `product_data/customer_timeline/canonical_readonly_*` и `audits/_inbox/*` уже игнорируются.
- Raw/test SQLite и JSON-отчёты не коммитить.

## Следующий шаг

- Дать D1/D7 финальный источник chat→customer для Wappi-чатов или расширить authoritative pairs так, чтобы часть сообщений смогла стать событиями клиента, затем повторить тестовую заливку и только после PASS решать боевой apply.
