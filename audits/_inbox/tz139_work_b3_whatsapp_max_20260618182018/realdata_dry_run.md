# Real Data Dry Run

Command class: local read-only dry-run, `apply=False`.

Source:

`/Users/dmitrijfabarisov/Projects/Mango analyse/all_whatsapp_chats.txt`

Existing timeline DB used only for read-time phone lookup:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_profiles_after_tail_20260613/customer_timeline.sqlite`

Result:

```json
{
  "validation_ok": true,
  "mode": "dry_run_preview",
  "source_unchanged": true,
  "write_status_counts": {},
  "links": {
    "unique_existing_phone_matches": 4515,
    "ambiguous_phone_matches": 0
  },
  "normalized_counts": {
    "customers": 35,
    "identity_links": 160066,
    "events": 40034,
    "bot_context_chunks": 40034,
    "conflicts": 0,
    "customer_id_mappings": 35
  },
  "summary": {
    "records_loaded": 40034,
    "messages_seen": 63584,
    "linked_by_phone": 39999,
    "session_only": 35,
    "writes_applied": 0,
    "source_unchanged": true,
    "safety_ok": true
  }
}
```

Separate dry-run with a non-existent probe DB confirmed that no SQLite file is created in dry-run mode.

Note: ambiguous phone matches are covered by unit/NEG fixture. The checked real WhatsApp phones against the selected latest timeline DB produced no ambiguous phone candidates.
