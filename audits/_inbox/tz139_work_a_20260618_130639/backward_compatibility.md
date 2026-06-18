# Backward Compatibility

## Preserved

- `stable_customer_id()` was not changed.
- Existing customer, event, opportunity, signal, chunk contracts remain available.
- Existing read APIs continue to read through the same store summary/list/search surfaces.
- Existing safety flags for live writes remain false.
- Existing tests outside customer_timeline pass in the full repo run.

## Added

- New SQLite table: `customer_id_mappings`.
- New store APIs:
  - `record_customer_id_mapping()`
  - `list_customer_id_mappings()`
- New safety flags:
  - `identity_conflicts_auto_merge=False`
  - `old_to_new_customer_id_mapping_required=True`
  - `brand_blocks_identity_merge=False`

## Compatibility Notes

- Existing DBs opened writable will bootstrap the new table with `CREATE TABLE IF NOT EXISTS`.
- Read-only opening of existing DBs still uses SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Mapping is additive and does not rewrite old event rows in-place.
