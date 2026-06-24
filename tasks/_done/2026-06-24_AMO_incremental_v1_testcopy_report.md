# AMO incremental v1 на тестовой копии customer_timeline

Дата: 2026-06-24
Ветка/worktree: `codex/tz-c-nightly-cursors` / `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`

## Статус

AMO-only incremental v1 выполнен на тестовой копии. Live-бот, AMO write, Tallanto, CRM write и боевая `customer_timeline` не трогались.

## Тестовая копия

- Источник: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`
- Тестовая копия: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors/product_data/customer_timeline/amo_incremental_testcopy_20260624_115228/customer_timeline.sqlite`
- Отчёт JSON: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors/product_data/customer_timeline/amo_incremental_testcopy_20260624_115228/amo_incremental_report.json`

## Endpoint'ы

Только GET:

- `/api/v4/leads` с `filter[updated_at][from]`
- `/api/v4/contacts` с `filter[updated_at][from]`
- `/api/v4/events` с `filter[created_at][from]` и `filter[type][]`

Endpoint notes не использовался, whitelist не расширялся.

## Курсоры

До:

```json
{
  "amo_contacts_updated_at": null,
  "amo_events_created_at": null,
  "amo_leads_updated_at": null
}
```

После:

```json
{
  "amo_contacts_updated_at": "2026-06-24T08:47:45+00:00",
  "amo_events_created_at": "2026-06-24T08:30:45+00:00",
  "amo_leads_updated_at": "2026-06-23T05:52:15+00:00"
}
```

Курсоры лежат в `ingestion_cursors`, не в `ingestion_runs`.

## Fetch / normalize

```json
{
  "amo_contacts_updated_at": {
    "fetched": 416,
    "normalized": 393,
    "skipped": {"ambiguous": 23}
  },
  "amo_leads_updated_at": {
    "fetched": 500,
    "normalized": 110,
    "skipped": {"ambiguous": 34, "unmatched": 356}
  },
  "amo_events_created_at": {
    "fetched": 500,
    "normalized": 81,
    "skipped": {"ambiguous": 43, "unmatched": 376}
  }
}
```

Типы событий из `/events`:

```json
{
  "fetched_type_counts": {
    "common_note_added": 90,
    "incoming_chat_message": 72,
    "incoming_mail": 67,
    "outgoing_chat_message": 111,
    "outgoing_mail": 160
  },
  "normalized_type_counts": {
    "incoming_chat_message": 10,
    "incoming_mail": 17,
    "outgoing_chat_message": 18,
    "outgoing_mail": 36
  }
}
```

`common_note_added` в этом срезе был получен из AMO, но не привязался к клиентам: все такие события попали в `unmatched/ambiguous`. Поэтому импортированных `note_body_missing` сейчас 0.

## Body status

```json
{
  "event_only": 81,
  "note_body_missing": 0
}
```

## Первый и повторный прогон

Первый прогон:

- `affected_customer_count`: 224
- `changed_customer_count`: 484
- AMO events: `81` событий, `162` созданных записи (event + raw chunk)

Повторный прогон с тем же входом:

- `changed_customer_count`: 0
- новых событий не создано
- `repeat_run_duplicates`: 13 из-за overlap и dedupe

## SQL-проверки

```text
amocrm_event|amo_note|81
amocrm_snapshot|amo_contact_snapshot|11390
amocrm_snapshot|amo_deal_stage|5390
amo_event_raw|0|1|81
amo_contacts_updated_at|2026-06-24T08:47:45+00:00
amo_events_created_at|2026-06-24T08:30:45+00:00
amo_leads_updated_at|2026-06-23T05:52:15+00:00
```

Все raw AMO event chunks: `allowed_for_bot=0`, `requires_manager_review=1`. `bot_safe_summary` не создавался.

Readback 10 свежих `amocrm_event`: для каждого события `record.entity_type/entity_id` имеет `identity_links` на тот же `customer_id`; `identity_link_ok=true` во всех 10/10 проверенных строках. ПДн и сырые id в отчёт не выводились.

## Примеры без ПДн

```json
[
  {"customer_id_masked": "cust...e336", "event_type": "amo_note", "source_system": "amocrm_event", "summary": "AMO outgoing_chat_message for lead; event only"},
  {"customer_id_masked": "cust...be2e", "event_type": "amo_note", "source_system": "amocrm_event", "summary": "AMO outgoing_mail for contact; event only"},
  {"customer_id_masked": "cust...878a", "event_type": "amo_note", "source_system": "amocrm_event", "summary": "AMO outgoing_chat_message for contact; event only"},
  {"customer_id_masked": "cust...7fa7", "event_type": "amo_note", "source_system": "amocrm_event", "summary": "AMO outgoing_chat_message for lead; event only"},
  {"customer_id_masked": "cust...4267", "event_type": "amo_contact_snapshot", "source_system": "amocrm_snapshot", "summary": null},
  {"customer_id_masked": "cust...c06b", "event_type": "amo_contact_snapshot", "source_system": "amocrm_snapshot", "summary": null},
  {"customer_id_masked": "cust...5861", "event_type": "amo_contact_snapshot", "source_system": "amocrm_snapshot", "summary": null},
  {"customer_id_masked": "cust...01f1", "event_type": "amo_contact_snapshot", "source_system": "amocrm_snapshot", "summary": null}
]
```

## Safety

```json
{
  "amo_write": false,
  "tallanto_write": false,
  "crm_write": false,
  "notes_endpoint_used": false,
  "bot_safe_summary_created": false,
  "test_copy_only": true
}
```

## Остаточные вопросы

- Notes endpoint не использовался. Если нужно полное тело заметки, нужен отдельный read-only whitelist для `/leads/notes` / `/contacts/notes`.
- В текущем срезе много `unmatched` по events. Это не склейка и не ошибка безопасности, но для большей полноты нужно улучшать связь AMO event entity -> существующий customer_id.
