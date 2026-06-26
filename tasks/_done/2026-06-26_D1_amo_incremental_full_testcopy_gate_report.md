# D1 AMO incremental full test-copy gate

Дата: 2026-06-26  
Repo: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`  
Base commit: `e877bb0`  

## Цель

Проверить AMO incremental на полной test-copy перед любым production apply.

Production apply НЕ выполнялся.

## Production DB, read-only

Боевая SQLite:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`

- SHA256 до/после gate: `ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`
- WAL: `0` bytes
- Production DB write: `0`

## Важный фикс в коде

Во время gate пойман контрактный баг: AMO event для lead нельзя писать с `opportunity_id=<AMO lead id>`, потому что `timeline_events.opportunity_id` должен ссылаться на внутренний `customer_opportunities.opportunity_id`.

Исправлено:

- строится индекс `customer_opportunities.source_id=<AMO lead id> -> opportunity_id`;
- `opportunity_id` для AMO lead-event ставится только при однозначном совпадении с тем же `customer_id`;
- если однозначной внутренней сделки нет, событие остаётся без `opportunity_id`, но с `customer_id`;
- добавлен регрессионный тест.

## Финальный test-copy прогон

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_customer_timeline_amo_incremental.py \
  --source-db "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite" \
  --out-root "/Users/dmitrijfabarisov/.mango_local/amo_incremental_full_gate_20260626_140309" \
  --page-limit 20 \
  --max-pages 25 \
  --sleep-sec 1.05 \
  --since "2026-06-25T00:00:00+03:00" \
  --summary-only
```

Артефакты:

- Test copy: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_full_gate_20260626_140309/customer_timeline.sqlite`
- Full report: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_full_gate_20260626_140309/amo_incremental_report.json`
- Journal: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_full_gate_20260626_140309/amo_incremental_journal.jsonl`
- Summary: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_full_gate_20260626_140309/summary_stdout.json`

## Fetch/result counts

AMO read-only GET:

- `/api/v4/leads`, `filter[updated_at][from]`: fetched `500`, normalized `193`
- `/api/v4/contacts`, `filter[updated_at][from]`: fetched `256`, normalized `249`
- `/api/v4/events`, `filter[created_at][from]`: fetched `500`, normalized `237`

Event types normalized:

- `common_note_added`: `6`
- `incoming_chat_message`: `30`
- `incoming_mail`: `23`
- `outgoing_chat_message`: `47`
- `outgoing_mail`: `131`

Mapping diagnostics for events:

- `mapped_before`: `142`
- `mapped_after_card_import`: `95`
- `ambiguous_before`: `13`
- `entity_not_in_fetched_cards`: `187`
- `fetched_card_but_no_link_after`: `63`

Opportunity mapping:

- `mapped`: `79`
- `missing_or_not_applicable`: `158`
- SQL check: AMO events with missing referenced internal opportunity: `0`

## Copy SQL checks

On test-copy after final run:

- `PRAGMA integrity_check`: `ok`
- `timeline_events`: `159392`
- `bot_context_chunks`: `127218`
- `amocrm_event`: `235`
- `amocrm_snapshot`: `16719`
- `amocrm_event` without `customer_id`: `0`
- `amocrm_event` with valid internal `opportunity_id`: `79`
- `amocrm_event` with invalid `opportunity_id`: `0`
- `amocrm_event` chunks safety:
  - `allowed_for_bot=0`
  - `requires_manager_review=1`
  - count `235`

Почему `amocrm_event` count `235`, а report normalized `237`: две входные AMO event-записи оказались дублями по `dedupe_key/source_id`. Импорт создал `235` уникальных timeline events и `235` manager-only chunks; `write_status_counts.duplicate=4` — это event+chunk для двух дублей. Gate condition здесь: нет unsafe write, нет событий без customer, нет битых opportunity refs, raw chunks manager-only, repeat `changed_customer_count=0`.

## Idempotency

Second run summary:

- `changed_customer_count`: `0`

## Safety

Report safety flags:

- AMO write: `false`
- Tallanto write: `false`
- CRM write: `false`
- notes endpoint used: `false`
- bot_safe_summary_created: `false`
- test_copy_only: `true`

## Tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_customer_timeline_amo_incremental.py \
  tests/test_customer_timeline_nightly_incremental.py \
  tests/test_customer_timeline_ingestion.py

26 passed
```

Full pytest:

```text
3658 passed, 5 skipped, 1 warning in 76.96s
```

## Gate decision

Formal full test-copy gate passed.

Production apply is still NOT done. Next step requires explicit approval and must start with a fresh production backup + SHA manifest, then apply the same runner to production or a controlled swap path selected by the owner.
