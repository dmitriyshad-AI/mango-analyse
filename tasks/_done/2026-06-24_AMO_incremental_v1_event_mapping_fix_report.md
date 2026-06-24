# AMO incremental v1: event -> customer mapping fix

Дата: 2026-06-24
Ветка/worktree: `codex/tz-c-nightly-cursors` / `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`

## Статус

Поправка выполнена и перемерена на тестовой копии. Боевую `customer_timeline`, AMO, Tallanto, CRM и live-бот не трогал.

## Что было причиной unmatched

Порядок `leads -> contacts -> events` в логике уже был, но `link_index` для событий загружался **до** импорта текущих leads/contacts. Поэтому события по контактам/сделкам, появившимся в этом же incremental-прогоне, не видели свежесозданные `identity_links`.

Проверка старого среза показала:

- `mapped_before`: 90 событий уже имели link до прогона;
- `mapped_after_card_import`: 113 событий могли сматчиться только после импорта карточек этого же прогона;
- `entity_not_in_fetched_cards`: 255 событий не имели link и их lead/contact не пришёл в `leads/contacts updated_at`-выборке;
- `ambiguous_before`: 42 события остались неоднозначными и не склеивались.

## Что изменено

- Импорт теперь двухфазный:
  1. fetch + import `leads/contacts`;
  2. reload `identity_links`;
  3. fetch + import `/events`.
- Для lead-карточек добавлен безопасный маппинг через embedded contact, если контакт уже имеет ровно один `customer_id`.
- Ambiguous не склеиваются.
- Unmatched остаются unmatched.
- Notes endpoint не использовался.

## Тестовая копия

- Источник: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`
- Новая тестовая копия: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors/product_data/customer_timeline/amo_incremental_testcopy_mapping_fix_20260624_124130/customer_timeline.sqlite`
- JSON-отчёт: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors/product_data/customer_timeline/amo_incremental_testcopy_mapping_fix_20260624_124130/amo_incremental_report.json`

## Endpoint'ы

Только GET:

- `/api/v4/leads` с `filter[updated_at][from]`
- `/api/v4/contacts` с `filter[updated_at][from]`
- `/api/v4/events` с `filter[created_at][from]`

`/leads/notes`, `/contacts/notes` и любые write endpoint'ы не использовались.

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
  "amo_contacts_updated_at": "2026-06-24T09:36:24+00:00",
  "amo_events_created_at": "2026-06-24T09:33:53+00:00",
  "amo_leads_updated_at": "2026-06-23T06:27:00+00:00"
}
```

## Новый выход `/events`

```json
{
  "fetched": 500,
  "normalized": 203,
  "skipped": {
    "ambiguous": 42,
    "unmatched": 255
  },
  "mapping_diagnostics_counts": {
    "mapped_before": 90,
    "mapped_after_card_import": 113,
    "entity_not_in_fetched_cards": 255,
    "ambiguous_before": 42
  }
}
```

До поправки в сопоставимом срезе было `normalized=81`; после поправки стало `normalized=203`.

Типы событий:

```json
{
  "fetched_type_counts": {
    "common_note_added": 83,
    "incoming_chat_message": 64,
    "incoming_mail": 62,
    "outgoing_chat_message": 120,
    "outgoing_mail": 171
  },
  "normalized_type_counts": {
    "incoming_chat_message": 8,
    "incoming_mail": 43,
    "outgoing_chat_message": 24,
    "outgoing_mail": 128
  }
}
```

## `common_note_added`

```json
{
  "fetched": 83,
  "normalized": 0,
  "diagnostics": {
    "entity_not_in_fetched_cards": 82,
    "ambiguous_before": 1
  }
}
```

Причина: почти все `common_note_added` указывают на lead/contact, который не пришёл в текущей `updated_at`-выборке leads/contacts и не имеет существующего `identity_link`. Один кейс уже был неоднозначным. Мы не делали per-entity GET и не трогали notes endpoint, поэтому это корректно осталось unmatched.

## Leads / contacts после правки

Leads:

```json
{
  "fetched": 500,
  "normalized": 169,
  "resolution_counts": {
    "direct_identity_link": 109,
    "embedded_contact_identity_link": 60,
    "direct_ambiguous": 36,
    "embedded_contact_ambiguous": 13,
    "unmatched": 282
  }
}
```

Contacts:

```json
{
  "fetched": 428,
  "normalized": 405,
  "resolution_counts": {
    "direct_identity_link": 139,
    "new_contact_identity": 266,
    "direct_ambiguous": 23
  }
}
```

Индекс AMO-связей: `12167 -> 12493` после card-import.

## Идемпотентность

Повторный прогон по тем же source-файлам:

- `changed_customer_count`: 0
- новых событий не создано
- dedupe/overlap сработали штатно

## SQL/readback

```text
amocrm_event|amo_note|203
amocrm_snapshot|amo_contact_snapshot|11402
amocrm_snapshot|amo_deal_stage|5449
amo_event_raw|0|1|203
amo_contacts_updated_at|2026-06-24T09:36:24+00:00
amo_events_created_at|2026-06-24T09:33:53+00:00
amo_leads_updated_at|2026-06-23T06:27:00+00:00
```

Readback 10 свежих `amocrm_event`: `identity_link_ok=true` в 10/10 проверенных строках. ПДн и сырые id в отчёт не выводились.

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

## Можно ли готовить apply на боевую

Да, после backup можно готовить отдельный controlled apply на боевую timeline с этой двухфазной логикой. Ограничение: `common_note_added` всё ещё почти полностью не импортируется, потому что его entities не попали в `updated_at`-выборку и notes endpoint запрещён. Это не blocker для безопасного apply событий chat/mail, но это явный остаточный gap полноты.
