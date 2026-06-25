# TZ FINAL Mango call increment production second append report

Дата: 2026-06-25
Ветка: `codex/mango-call-increment`
База перед началом: `98d88e3`.

## Что выполнено

По явному согласию Дмитрия выполнен второй боевой append-долив готовых Mango-разборов в production customer_timeline.

Границы соблюдены:

- AMO/Tallanto/CRM: не трогались.
- ASR/Analyze/download: не запускались.
- `stable_runtime`: не менялся.
- Production timeline изменялась append-записью событий/chunks/conflicts через `TimelineImportService`.
- Боевой cursor не продвигался: `cursor_after=[]`.

## Backup

Backup production timeline перед записью:

`/Users/dmitrijfabarisov/Projects/Mango_release_venue_autonomy/_external_handoffs/mango_call_increment_prod_second_20260625/customer_timeline_prod_backup_before_mango_increment_second_20260625.sqlite`

Backup `PRAGMA quick_check`: `ok`.

## Source inventory

Окно: `2026-06-19T00:00:00+00:00` — `2026-06-26T00:00:00+00:00`.

- Всего строк: 1036
- `done` + валидный `analysis_json`: 1033
- pending/processing/not-ready: 3

Сравнение с предыдущим боевым проходом: было `done=873`, стало `1033`, прирост готовых source rows = 160.

## Precheck production DB

До записи:

- `quick_check`: `ok`
- customer_identities: 17851
- identity_links: 84924
- timeline_events: 158552
- mango_processed_summary events: 72835
- `source_system='mango_call'`: 0
- cursor_before: `[]`

## Producer

- rows_read/selected/events_written: 1033 / 1033 / 1033
- source_counts: `{'call_records': 1033}`
- identity_resolution: `{'ambiguous': 69, 'strong_unique': 453, 'unmatched': 511}`
- call_type_counts: `{'existing_client_progress': 22, 'non_conversation': 261, 'sales_call': 506, 'service_call': 239, 'technical_call': 5}`

## Production append import

- accepted/rejected: 1033 / 0
- normalized_counts: `{'artifacts': 0, 'bot_context_chunks': 311, 'conflicts': 649, 'customer_id_mappings': 0, 'customers': 0, 'events': 1033, 'identity_links': 0, 'opportunities': 0, 'signals': 0}`
- write_status_counts: `{'created': 310, 'duplicate': 1683}`
- cursor_policy: `not_updated_source_in_motion`
- run_id: `timeline_ingestion_run:c8e85c955725da2477776da3def8ca94`

## Repeat check

Повторный import того же JSONL:

- accepted/rejected: 1033 / 0
- write_status_counts: `{'duplicate': 1993}`
- новых событий/chunks/customer/links не создано; все записи ушли в duplicate.

## Final post-check

После append + repeat:

- `PRAGMA quick_check`: `ok`
- source_systems for mango_call: `['mango_processed_summary']`
- `source_system='mango_call'`: 0
- customer_identities: 17851
- identity_links: 84924
- timeline_events: 158715
- mango_processed_summary events: 72998
- provider events: 1036
- provider dedupe distinct: 1036
- match_status: `{'ambiguous': 69, 'strong_unique': 453, 'unmatched': 514}`
- provider chunks: 311
- chunks `allowed_for_bot != 0`: 0
- chunks `requires_manager_review != 1`: 0
- non_conversation events/chunks: 261 / 0
- cursor_after: `[]`

## Edge case: duplicate source_call_id in live source

Во время второго прохода найден важный edge case:

- Текущий producer JSONL: 1033 source_id.
- Production provider events после второго append: 1036.
- `extra_in_prod_not_current_jsonl`: `['provider:27115358400', 'provider:27115998244', 'provider:27117440506']`.

Причина: у трёх `source_call_id` второй ряд стал `done` позже. Ранее одиночная строка была импортирована как base `provider:<id>`, а когда появились обе строки, producer стал использовать suffixed source_id для обеих. Это создало 3 исторических base+suffix semantic duplicates по original_call_id.

Список problematic base+suffix groups зафиксирован в ignored артефакте:

`_external_handoffs/mango_call_increment_prod_second_20260625/semantic_duplicate_original_call_ids_after_second_import.json`

Что сделано для будущих проходов: producer исправлен, теперь он считает повторяющиеся `source_call_id` по всем строкам окна, включая pending/not-done, поэтому source_id не будет менять форму при дозревании соседней строки.

Важно: существующие 3 base-события я не удалял и не правил без отдельного разрешения Дмитрия.

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

Второй боевой append выполнен и проверен:

- Добавлено новых production timeline events относительно precheck: 163.
- Mango provider events final: 1036.
- Создано новых chunks относительно precheck: 43.
- Новых customers/identity_links/customer mappings: 0.
- `mango_call` source_system не появился.
- `allowed_for_bot` violations: 0.
- `non_conversation` chunks: 0.
- Pending/not-ready строк осталось: 3.

Следующий безопасный шаг: дождаться оставшихся 3 pending, сделать ещё один маленький pass уже с исправленным producer, затем отдельно решить вопрос с 3 semantic duplicate base-событиями и пересборкой customer profiles / bot-safe summaries.
