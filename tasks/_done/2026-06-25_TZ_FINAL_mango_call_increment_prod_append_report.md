# TZ FINAL Mango call increment production append report

Дата: 2026-06-25
Ветка: `codex/mango-call-increment`
Код producer/normalizer: commit `e01304f`; отчётный commit предыдущего dry-run: `e082553`.

## Что выполнено

По явному согласию Дмитрия выполнен боевой append-долив готовых Mango-разборов в production customer_timeline.

Границы соблюдены:

- AMO/Tallanto/CRM: не трогались.
- ASR/Analyze/download: не запускались.
- `stable_runtime`: не менялся.
- Production timeline изменялась только append-записью событий/chunks/conflicts через `TimelineImportService`.
- Боевой `ingestion_cursor` не продвигался, потому что источник ещё в движении. Store создал пустую таблицу `ingestion_cursors`, но записей cursor для `mango_processed_summary` нет.

## Backup

Backup production timeline снят через SQLite backup API перед записью:

`/Users/dmitrijfabarisov/Projects/Mango_release_venue_autonomy/_external_handoffs/mango_call_increment_prod_20260625/customer_timeline_prod_backup_before_mango_increment_20260625.sqlite`

Backup `PRAGMA quick_check`: `ok`.

## Source inventory перед доливом

Окно: `2026-06-19T00:00:00+00:00` — `2026-06-26T00:00:00+00:00`.

- Всего строк: 1036
- `done` + валидный `analysis_json`: 864
- pending/processing/not-ready: 172

Pending-остаток не импортировался и должен прийти следующим проходом.

## Precheck production DB

До записи:

- `quick_check`: `ok`
- customer_identities: 17851
- identity_links: 84924
- timeline_events: 157679
- mango_processed_summary events: 71962
- `source_system='mango_call'`: 0
- source_systems for mango_call: `['mango_processed_summary']`
- ingestion_cursors table existed before: False

## Producer

- rows_read/selected/events_written: 873 / 873 / 873
- source_counts: `{'call_records': 873}`
- identity_resolution: `{'ambiguous': 62, 'strong_unique': 387, 'unmatched': 424}`
- call_type_counts: `{'existing_client_progress': 18, 'non_conversation': 220, 'sales_call': 430, 'service_call': 201, 'technical_call': 4}`
- source_system: `mango_processed_summary`

## Production append import

- accepted/rejected: 873 / 0
- normalized_counts: `{'artifacts': 0, 'bot_context_chunks': 268, 'conflicts': 548, 'customer_id_mappings': 0, 'customers': 0, 'events': 873, 'identity_links': 0, 'opportunities': 0, 'signals': 0}`
- write_status_counts: `{'created': 1689}`
- cursor_policy: `not_updated_source_in_motion`
- run_id: `timeline_ingestion_run:6a09ca6ad80c9cb33fbef1027421d0a5`

## Repeat check

Повторный import того же JSONL выполнен как idempotency-check:

- accepted/rejected: 873 / 0
- write_status_counts: `{'duplicate': 1689}`
- новых событий/chunks/customer/links не создано; все записи ушли в duplicate.

## Final post-check

После append + repeat:

- `PRAGMA quick_check`: `ok`
- source_systems for mango_call: `['mango_processed_summary']`
- `source_system='mango_call'`: 0
- customer_identities: 17851
- identity_links: 84924
- timeline_events: 158552
- mango_processed_summary events: 72835
- provider events: 873
- provider dedupe distinct: 873
- match_status: `{'ambiguous': 62, 'strong_unique': 387, 'unmatched': 424}`
- provider chunks: 268
- chunks `allowed_for_bot != 0`: 0
- chunks `requires_manager_review != 1`: 0
- non_conversation events/chunks: 220 / 0
- cursor_after for `mango_processed_summary`: `[]`

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

## Итог

Боевой append-долив выполнен и проверен:

- Добавлено 873 Mango-события в production timeline.
- Создано 268 manager-only chunks.
- Новых customers/identity_links/customer mappings: 0.
- `mango_call` source_system не появился.
- `allowed_for_bot` violations: 0.
- `non_conversation` chunks: 0.
- Pending/not-ready строк осталось: 172.

Следующий безопасный шаг: после завершения текущей обработки звонков сделать следующий incremental pass по тем же правилам, чтобы добрать оставшиеся pending, и затем отдельно решить, когда пересобирать customer profiles / bot-safe summaries.
