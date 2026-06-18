# TZ139 Work A Implementation Notes

Дата: 2026-06-18
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline`
Branch: `codex/tz139-customer-timeline`

## Scope

Выполнены Step 0 и Work A из staged-плана TZ139:

- read-only baseline по customer_timeline;
- entity resolution для phone-union без silent merge;
- защита семейного Tallanto phone;
- old -> new customer_id mapping;
- multi-valued brand history без brand-blocking identity;
- canonical importer защита от `phone -> row` overwrite.

Work B-F не выполнялись. Дизайн миграций C/F не трогался.

## Implementation

- `store.py`: добавлена таблица `customer_id_mappings`, индексы, `record_customer_id_mapping()`, `list_customer_id_mappings()`, счётчик в `summary()`.
- `ingestion.py`: добавлен deterministic resolver перед записью batches. При apply он читает существующие identity links из native timeline DB, чтобы AMO/Mango/Tallanto imports в разных CLI-прогонах сходились по телефону.
- `canonical_readonly_import.py`: индекс клиентов больше не теряет строки с одним телефоном; family phone с несколькими Tallanto students остаётся split и конфликтным.
- `safety.py`: safety contract явно фиксирует `identity_conflicts_auto_merge=False`, `old_to_new_customer_id_mapping_required=True`, `brand_blocks_identity_merge=False`.
- Тесты добавлены в `tests/test_customer_timeline_store.py`, `tests/test_customer_timeline_ingestion.py`, `tests/test_customer_timeline_canonical_readonly_import.py`.

## NEG Coverage

- Family Tallanto phone: два student ID на одном телефоне остаются двумя customer, создаётся conflict, phone links не схлопываются.
- Canonical family phone: звонки не теряются на dedupe, events создаются для обоих split customers с `ambiguous` match.
- Generic cross-run AMO -> Mango: второй import подхватывает existing store identity и не создаёт второго customer.
- Split mapping: один legacy phone-level old id может вести к нескольким split new customer_id.
- Brand-only difference: не создаёт конфликт и не делит identity; `brands` сохраняет историю.

## Exclusions

- Source DB snapshots and real raw measurements not used.
- No AMO/Tallanto/CRM writes.
- No ASR, Resolve+Analyze, stable_runtime writes, or heavy batch runs.
