# D4 Email Timeline Bridge Report

## Result

Implemented and ran the test email+calls timeline bridge using fresh-only relink, not union.

## Files Changed

- `src/mango_mvp/customer_timeline/ingestion.py`
- `src/mango_mvp/customer_timeline/store.py`
- `scripts/import_mail_bridge_to_customer_timeline.py`
- `tests/test_customer_timeline_ingestion.py`
- `tests/test_customer_timeline_import_cli.py`
- `docs/worktrees_registry.md`
- `tasks/_running/2026-06-21_TZ_D4_most_pisma_v_pamyat.md`

## Test DB

- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_email_bridge_seed_full5_20260621/customer_timeline.sqlite`
- Full report: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_email_bridge_seed_full5_20260621/mail_fresh_relink_bridge_report.json`

## Counts

- Seed canonical timeline: 16,239 customers, 116,086 events, 77,482 bot chunks.
- Email records read: 30,093.
- Resolved by fresh relink: 22,491.
- Pending attribution: 7,602.
- New bridge email events: 22,491.
- New pending attribution conflicts: 7,602.
- `PRAGMA quick_check`: ok.
- Duplicate event groups: 0.
- Mail/channel bot chunks with `allowed_for_bot=1`: 0.
- Full repeat pass created 0 new rows.

## Tests

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_store.py tests/test_customer_timeline_import_cli.py`

Result: 39 passed.

## Notes

- Inline `customer_id` from JSONL is not used.
- `already_linked` rows are resolved only when `old_customer_id_hash` maps to a real Tallanto ID in the fresh identity map.
- `linked` rows are resolved by `tallanto_id_hash`.
- `unmatched` and unresolved hashes become `pending_attribution`.
- Existing seed customers are not overwritten by mail-derived minimal identity records.
- Earlier interrupted test artifact folders were left untouched under ignored `product_data/customer_timeline/`.

