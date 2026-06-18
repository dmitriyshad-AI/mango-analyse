# Real-data dry-run

Команда была выполнена через Python API `run_telegram_export_import(...)` с `apply=False`.

- source messages: `/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01`
- identity sidecar: `/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01_max`
- lookup DB: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_20260521_v5/customer_timeline.sqlite`
- allowed root: `/Users/dmitrijfabarisov/Projects`
- writes applied: 0
- source_unchanged: true
- validation_ok: true
- telegram_api_called: false

Counts:

```json
{
  "source": {
    "dialogs_total": 1653,
    "messages_total": 13223,
    "telegram_api_called": false
  },
  "counters": {
    "dialogs": 1653,
    "messages": 13223,
    "skipped": 1223,
    "groups": 364,
    "dialog_events": 969,
    "linked_by_phone": 3740,
    "linked_by_username": 8029,
    "session_only": 2180,
    "imported": 12000,
    "duplicates": 0,
    "bad_jsonl_rows": 0,
    "ambiguous_dialogs": 0,
    "unmatched_dialogs": 780
  },
  "links": {
    "unique_existing_phone_matches": 189,
    "ambiguous_phone_matches": 0,
    "unique_existing_username_matches": 0,
    "ambiguous_username_matches": 0,
    "existing_duplicate_source_ids": 0,
    "message_match_counts": {
      "dialogs_total": 969,
      "strong_unique_dialogs": 189,
      "ambiguous_dialogs": 0,
      "unmatched_dialogs": 780
    }
  },
  "normalized_counts": {
    "customers": 9022,
    "identity_links": 35769,
    "opportunities": 969,
    "events": 12969,
    "bot_context_chunks": 12000,
    "artifacts": 0,
    "signals": 0,
    "conflicts": 0
  }
}
```

Пояснение: `customers` в dry-run preview считает planned upsert per source record; фактическая идемпотентность по одному диалогу покрыта тестом repeat import. На canonical lookup в этой версии не найдено ambiguous phone candidates для Telegram sidecar; family/shared phone path покрыт NEG-тестом `test_ambiguous_phone_match_is_counted_without_first_match_merge`.
