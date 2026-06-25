# TZ FINAL Mango call increment extended test-copy run

Дата: 2026-06-25
Ветка: `codex/mango-call-increment`
Код producer/runner: commit `e01304f`
Канон ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-25_TZ_FINAL_increment_zvonkov_dlya_D4.md`

## Решение по ТЗ

С ТЗ согласен. Выполнил именно расширенный прогон на тест-копии: код не менял, боевую timeline не трогал, AMO/Tallanto/CRM не трогал, ASR/Analyze/download не запускал.

## Окно и источники

Окно: `2026-06-19T00:00:00+00:00` — `2026-06-26T00:00:00+00:00`.

Источники: package-local `call_records` из `mango_update_after_20260619_*` ... `mango_update_after_20260625_20260625_v1`.

Важно: в одном package root найден stale backup SQLite `*.before_ra_stale_claim_recovery_*.sqlite`; чтобы не читать мусорный backup, прогон выполнен по явным рабочим DB из `RA_FINAL_SUMMARY.json`, а не по широкому glob `--package-root`.

## Тестовая копия

Копия снята через SQLite backup API, не `cp`:

`/Users/dmitrijfabarisov/Projects/Mango_release_venue_autonomy/_external_handoffs/mango_call_increment_extended_20260625/customer_timeline_testcopy.sqlite`

Рабочая папка артефактов:

`/Users/dmitrijfabarisov/Projects/Mango_release_venue_autonomy/_external_handoffs/mango_call_increment_extended_20260625`

Initial `PRAGMA quick_check`: `ok`.
Final `PRAGMA quick_check`: `ok`.

## Source inventory

Первый read-only inventory:

- всего строк в окне: 1036
- `done` + валидный `analysis_json`: 784
- pending/processing/not-ready: 252

Второй inventory после паузы:

- всего строк в окне: 1036
- `done` + валидный `analysis_json`: 798
- pending/processing/not-ready: 238
- новых `done` за паузу: 14

## Producer first run

- rows_read/selected/events_written: 784 / 784 / 784
- source_counts: `{'call_records': 784}`
- identity_resolution: `{'ambiguous': 58, 'strong_unique': 355, 'unmatched': 371}`
- call_type_counts: `{'existing_client_progress': 18, 'non_conversation': 198, 'sales_call': 384, 'service_call': 181, 'technical_call': 3}`

## Import first run

- source_rows_selected: 784
- changed_customer_count: 233
- affected_customer_count: 233
- normalized_counts: `{'artifacts': 0, 'bot_context_chunks': 246, 'conflicts': 487, 'customer_id_mappings': 0, 'customers': 0, 'events': 784, 'identity_links': 0, 'opportunities': 0, 'signals': 0}`
- write_status_counts: `{'created': 1517}`

## Обязательная post-проверка после первого импорта

- `SELECT DISTINCT source_system ... event_type='mango_call'`: `['mango_processed_summary']`
- `source_system='mango_call'`: 0
- provider events: 784
- dedupe distinct: 784
- match_status: `{'ambiguous': 58, 'strong_unique': 355, 'unmatched': 371}`
- chunks: 246
- chunks `allowed_for_bot != 0`: 0
- chunks `requires_manager_review != 1`: 0
- non_conversation events/chunks: 198 / 0

## Concurrency test

После паузы producer перечитал источники read-only. За это время `done_valid` вырос с 784 до 798 (+14).

Second runner на той же тест-копии:

- source_rows_selected: 15 (overlap + новые done)
- changed_customer_count: 5
- affected_customer_count: 6
- normalized_counts: `{'artifacts': 0, 'bot_context_chunks': 4, 'conflicts': 10, 'customer_id_mappings': 0, 'customers': 0, 'events': 15, 'identity_links': 0, 'opportunities': 0, 'signals': 0}`
- write_status_counts: `{'created': 27, 'duplicate': 2}`
- фактический прирост unique provider events: 14

Итог после second run:

- provider events: 798
- dedupe distinct: 798
- source_systems: `['mango_processed_summary']`
- `source_system='mango_call'`: 0
- final match_status: `{'ambiguous': 59, 'strong_unique': 360, 'unmatched': 379}`
- final chunks: 249
- final chunks `allowed_for_bot != 0`: 0
- final chunks `requires_manager_review != 1`: 0
- final non_conversation events/chunks: 205 / 0
- customers/identity_links count stayed: 17851 / 84924

## 10 обезличенных примеров

1. `provider:27060410848` | 2026-06-19T06:42:26+00:00 | sales_call | unmatched | customer_id_present=False | phone=+79***87
2. `provider:27060573346` | 2026-06-19T06:53:20+00:00 | sales_call | unmatched | customer_id_present=False | phone=+79***90
3. `provider:27060610654` | 2026-06-19T06:55:48+00:00 | non_conversation | strong_unique | customer_id_present=True | phone=+79***91
4. `provider:27060640355` | 2026-06-19T06:57:48+00:00 | sales_call | unmatched | customer_id_present=False | phone=+79***48
5. `provider:27060768671` | 2026-06-19T07:06:38+00:00 | service_call | unmatched | customer_id_present=False | phone=+74***27
6. `provider:27060900563` | 2026-06-19T07:13:17+00:00 | service_call | strong_unique | customer_id_present=True | phone=+79***80
7. `provider:27060931090` | 2026-06-19T07:15:07+00:00 | sales_call | strong_unique | customer_id_present=True | phone=+79***22
8. `provider:27060984617` | 2026-06-19T07:18:21+00:00 | sales_call | strong_unique | customer_id_present=True | phone=+79***52
9. `provider:27061017948` | 2026-06-19T07:20:31+00:00 | sales_call | unmatched | customer_id_present=False | phone=+79***58
10. `provider:27061053818` | 2026-06-19T07:22:23+00:00 | service_call | unmatched | customer_id_present=False | phone=+79***27

## Вывод

Formal pass на тест-копии:

- `source_system` не расползся: только `mango_processed_summary`.
- `mango_call` не появился.
- Клиенты и identity_links из воздуха не созданы.
- `ambiguous/unmatched` остались без `customer_id` и ушли в pending attribution.
- `non_conversation` не дал chunks.
- Все chunks manager-only: `allowed_for_bot=0`, `requires_manager_review=1`.
- Concurrency-повтор поймал новые `done` и не задублировал старые events (`dedupe distinct = provider events`).

Вердикт «в прод» НЕ выносится. Нужен регрейд Claude #1 по сырью и отдельная отмашка Дмитрия на боевой долив.
