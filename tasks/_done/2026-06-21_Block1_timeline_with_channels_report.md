# Block1 — timeline with channels night run

Дата: 2026-06-20/21  
Ветка/worktree: `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline` (`codex/tz139-customer-timeline-integrate`)  
Live-write: не запускался. AMO/Tallanto/IMAP/ASR/Resolve+Analyze не вызывались.

## Итоговый артефакт

Новая БД:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/customer_timeline.sqlite`

Папка:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/`

Текущая БД `canonical_readonly_20260619_migration_20260618_002_copy` не перезаписывалась.

## Источники

- Base timeline: read-only источники из `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/...`, явными путями.
- Full call analysis: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`
- Telegram: `/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01/`
- Telegram identity: `/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01_max/`
- WhatsApp: `/Users/dmitrijfabarisov/Projects/Mango analyse/all_whatsapp_chats.txt`
- UNPK Telegram HTML/TDesktop: `skipped_incompatible_format`; конвертация ночью не делалась.
- Max: `blocked/no-archive`; AMO/Wappi не вызывались.
- Email: `blocked/metadata-only`; IMAP не вызывался.

Примечание: ночное ТЗ называет `local_vm_2024-04-01` Фотон-экспортом; старый ТЗ-13 называл этот же экспорт УНПК. В ночном прогоне выбран свежий контракт ТЗ: `brand=foton`. Конфликт нужно сверить утром по сырью.

## Команды/отчёты

- Base build: `build_stdout.json`, `import_report.json`, `coverage_report.json`
- Telegram import: `import_telegram_local_vm_2024-04-01.json`
- WhatsApp import: `import_whatsapp_all_chats.json`
- Derived signals: `derive_signals_apply.json`, `derive_signals_apply_repeat.json`

## Счётчики БД

- `customer_identities`: 19 736
- `identity_links`: 240 804
- `customer_opportunities`: 4 980
- `timeline_events`: 179 212
- `bot_context_chunks`: 137 223
- `timeline_conflicts`: 4 414
- `customer_id_mappings`: 20 999
- `derived_signals`: 2 550
- `ingestion_runs`: 3
- `timeline_event_fts`: 179 212
- `bot_context_chunk_fts`: 137 223

Каналы:

- Telegram events: 12 969
- Telegram bot_context_chunks: 12 000
- WhatsApp events: 40 034
- WhatsApp bot_context_chunks: 40 034

## Импорты каналов

Telegram:

- validation: true
- accepted: 12 000
- dialogs: 1 653
- messages: 13 223
- ambiguous_dialogs: 26
- unmatched_dialogs: 780
- linked_by_phone: 3 740
- linked_by_username: 8 029
- session_only: 2 180

WhatsApp:

- validation: true
- accepted: 40 034
- source size: 17 036 610 bytes
- normalized conflicts: 3 787
- normalized customer_id_mappings: 15 298

## ПДн / bot-safe gate

- `bad_channel_chunks`: 0
- `bot_safe_channel_fts_hits`: 0
- Все Telegram/WhatsApp channel chunks имеют `allowed_for_bot=False` и `requires_manager_review=True`.

Ограничение: физически каналовый текст всё ещё попадает в FTS-таблицу, но bot-safe путь поиска фильтрует `allowed_for_bot=True`; прямых bot-safe hits по каналам нет. Для физического исключения каналов из FTS нужен отдельный дневной ТЗ.

## Derived signals

Первый apply:

- customers: 19 736
- `hot_lead_silent_7d`: 2 550
- status: `stale`: 2 550
- write_status: `created`: 2 550

Повторный apply:

- `hot_lead_silent_7d`: 2 550
- write_status: `duplicate`: 2 550

Идемпотентность подтверждена: повторный прогон не создаёт новые строки.

## Performance / остаточные риски

- Base build завершился штатно.
- Telegram import занял существенно дольше базовой записи из-за FTS-синхронизации на channel import.
- WhatsApp import был главным узким местом ночи: около 40+ минут на одно ядро.
- В коде есть FTS-в-конце для base builder, но нет такого режима для Telegram/WhatsApp importers. Это главный кандидат на дневную оптимизацию.
