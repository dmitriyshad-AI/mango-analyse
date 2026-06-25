# TZ v2 Mango call increment report

Дата: 2026-06-25
Ветка: `codex/mango-call-increment`
База worktree: `codex/release-venue-autonomy` / `4caa5eb`

## Что сделано

- Реализован read-only producer `scripts/build_mango_call_timeline_increment.py` для готовых Mango analysis JSON из `canonical_calls` и package-local `call_records`.
- Сохранён существующий `source_system=mango_processed_summary`, `event_type=mango_call`; новый `source_system=mango_call` не создавался.
- Producer резолвит телефон против существующих `identity_links` timeline:
  - ровно один existing customer -> `strong_unique` + `customer_id`;
  - несколько existing customer / unsafe link -> `ambiguous`;
  - нет phone/link -> `unmatched`.
- `MangoCallSummaryNormalizer` получил strict increment-режим: не создаёт `customer_identities`/`identity_links` из телефона, если identity уже решена producer-ом.
- `ambiguous`/`unmatched` пишутся как события без клиента + `pending_attribution`, а не клеятся к одному клиенту.
- `non_conversation` не создаёт содержательную summary/chunk/signal.
- `nightly_incremental` использует один Mango-normalizer и для load, и для import, чтобы `changed_customer_ids` считались тем же способом, что запись.

## Pilot на тестовой копии

Исходная prod timeline открывалась read-only; копия снята через SQLite backup API:

`_external_handoffs/mango_call_increment_20260625_pilot/customer_timeline_testcopy.sqlite`

Пакет: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_update_after_20260624_20260625_v1`
Ограничение: `limit=50`, только `analysis_status='done'` + валидный `analysis_json`.

Producer:

- rows_read: 211
- rows_selected/events_written: 50
- source_counts: `call_records=50`
- identity_resolution: `strong_unique=27`, `ambiguous=3`, `unmatched=20`
- call_type: `sales_call=24`, `service_call=8`, `non_conversation=18`
- writes_amo/writes_tallanto/writes_crm/runs_asr/runs_analyze: false

First import на копии:

- events: 50
- bot_context_chunks: 16
- conflicts: 26
- customers: 0
- identity_links: 0
- customer_id_mappings: 0
- changed_customer_count: 23
- source_errors: []

Repeat import:

- changed_customer_count: 0
- selected overlap rows: 1
- write_status_counts: `duplicate=3`
- новых событий/клиентов не создано

DB validation after pilot:

- `PRAGMA quick_check = ok`
- `source_system='mango_call'` events: 0
- provider pilot events: 50
- match_status: `strong_unique=27`, `ambiguous=3`, `unmatched=20`
- events with `customer_id IS NULL`: 23
- provider chunks: 16
- chunks `allowed_for_bot != 0`: 0
- chunks `requires_manager_review != 1`: 0
- pending_attribution conflicts for increment: 23

## Tests

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_contracts.py tests/test_customer_timeline_store.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_productization_call_processing_readiness.py tests/test_tz19_calls_review_table.py tests/test_mango_call_timeline_increment.py
```

Результат: `128 passed`.

## Audit pack

`audits/_inbox/mango_call_increment_tz_v2_20260625172549/`

В pack есть `pilot_summary.json` с обезличенными счётчиками; raw JSONL/SQLite лежат только в ignored `_external_handoffs/` и не предназначены для git.

## Что не делалось

- Боевая customer_timeline БД не изменялась.
- AMO/Tallanto/CRM не вызывались на запись.
- ASR/Analyze/Mango download не запускались.
- Профили и bot-safe summary не пересобирались.
- Вердикт о production-доливе не выносился: нужен регрейд Claude #1 по сырью и отдельная отмашка.
