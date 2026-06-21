# ТЗ-1 — ingest апрельского Telegram-экспорта в тестовую customer_timeline

Дата: 2026-06-21  
Ветка: `codex/tz1-telegram-aprel`  
Seed: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`  
Тестовая БД: `/Users/dmitrijfabarisov/Projects/Mango_tz1_telegram_aprel/product_data/customer_timeline/canonical_readonly_telegram_aprel_testcopy_20260621T001407Z/customer_timeline.sqlite`

## Что Сделано

- Взял экспорт `/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01_max`.
- Бренд задан явно по указанию Дмитрия: `unpk`.
- Загрузка выполнена только в тестовую копию SQLite, не в боевую базу.
- `source_system=telegram_history`, все Telegram chunks: `allowed_for_bot=0`, `requires_manager_review=1`.
- Добавлен защитный запрет `telegram_history` в `BOT_FORBIDDEN_SOURCE_SYSTEMS`.
- Привязка оставлена только по телефону: primary из `dialogs.jsonl`/`crm_contacts.csv`, secondary из текста сообщения.
- Username больше не используется для атрибуции, чтобы не было самосклейки на повторном прогоне.
- Несматч и общий/спорный телефон пишутся в `pending_attribution` / `telegram_identity_ambiguous`, без создания синтетического клиента.

## Источник

- `dialogs.jsonl`: 1653 строк.
- `crm_contacts.csv`: 1653 строки, телефоны у 725 строк.
- `messages.jsonl`: 7709 строк.
- В `summary.json`: `total_messages=7339`; даты сообщений с `2024-04-01T06:46:28+00:00` по `2026-04-15T06:47:40+00:00`.
- Внешнее ТЗ ожидало `13223` сообщения; фактический файл содержит меньше. Импортировал только фактический read-only источник.

## Счётчики Apply

- Accepted records: 7085, rejected: 0.
- Диалогов с импортируемыми user-сообщениями: 869.
- Strong unique dialogs: 201.
- Ambiguous dialogs: 35.
- Unmatched dialogs: 633.
- Pending attribution dialogs: 668.
- Сообщений с телефоном из `dialogs.jsonl`/`crm_contacts.csv`: 1816.
- Сообщений с телефоном из текста: 154.
- Сообщений с несколькими телефонами в тексте: 8, оставлены pending.

## SQLite После Apply

- `PRAGMA quick_check`: `ok`.
- `telegram_history` events: 1377.
- `telegram_message` events: 1230.
- `telegram_dialog` events: 147.
- `telegram_history` bot chunks: 1230.
- `allowed_for_bot=1` среди `telegram_history`: 0.
- `requires_manager_review=0` среди `telegram_history`: 0.
- События не `unpk`: 0.
- Синтетические клиенты из Telegram: 0.
- `pending_attribution` conflicts: 5667.
- `telegram_identity_ambiguous` conflicts: 188.

## Идемпотентность

Повторный `--apply` на той же тестовой БД:

- `imported=0`.
- `duplicates=7085`.
- `created`: 0 новых строк.
- `duplicate`: 11226.
- `updated`: 1975.
- Повтор не создал новых событий/chunks/customers.

## Безопасные Примеры Ленты

Тексты, имена, телефоны и email не выводились. `customer_ref` — короткий sha256 от customer_id.

| customer_ref | event_at | direction | phone_source | resolution | text_length |
|---|---|---|---|---|---:|
| `b20511a1ae5e` | 2024-04-02T21:29:39+00:00 | inbound | dialogs | strong_unique | 49 |
| `91a3909a98a2` | 2024-04-03T11:32:19+00:00 | inbound | dialogs | strong_unique | 50 |
| `5bdcd589fda6` | 2024-04-12T08:40:32+00:00 | inbound | message_text | strong_unique | 138 |
| `058f65e8b893` | 2024-04-15T01:04:13+00:00 | inbound | dialogs | strong_unique | 38 |
| `85d7ccc6256d` | 2024-04-25T09:03:17+00:00 | inbound | message_text | strong_unique | 88 |

## Локальные Артефакты

- Audit pack: `audits/_inbox/tz1_telegram_aprel_20260621_0014/`.
- Dry-run: `audits/_inbox/tz1_telegram_aprel_20260621_0014/dry_run_report.json`.
- Apply: `audits/_inbox/tz1_telegram_aprel_20260621_0014/apply_report.json`.
- Idempotency: `audits/_inbox/tz1_telegram_aprel_20260621_0014/idempotency_report.json`.
- Safe summary: `audits/_inbox/tz1_telegram_aprel_20260621_0014/safe_summary.json`.

Эти audit/raw артефакты находятся в ignored-папках и не предназначены для git.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_import_telegram_export_to_timeline.py tests/test_customer_timeline_ingestion.py::test_mail_and_channel_sources_reject_allowed_for_bot_true`
  - `15 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3482 passed, 5 skipped, 1 warning in 58.16s`

## NEG

- Общий телефон не склеен: 35 ambiguous dialogs ушли в конфликт, не в клиента.
- Несматч не создаёт клиента: `telegram_customers_created=0`.
- `allowed_for_bot=1` для `telegram_history` отклоняется тестом.
- Повторный запуск не создаёт новых строк.
- Бренд не угадывался: везде `unpk`, событий не `unpk` = 0.

## Остаточный Риск

- 633 диалога остались unmatched: это ожидаемо, потому что нет уникального телефона/одного телефона в тексте.
- Полный экспорт содержит период 2024-04-01…2026-04-15, хотя задача называется «апрельский экспорт».
- В `apply_report.json` старого формата нет новых полей `existing_duplicate_message_events` и `existing_duplicate_pending_conflicts`; они есть в финальном `idempotency_report.json` после доработки измерителя.
