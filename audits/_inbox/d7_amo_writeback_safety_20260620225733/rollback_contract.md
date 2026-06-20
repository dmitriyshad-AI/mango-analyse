# Snapshot / rollback

- Snapshot format: `pre_write_snapshot.jsonl/csv`.
- Existing lead fields retained: `lead_id`, `old_value_sha256`, `new_value_sha256`.
- New generic fields: `entity_type`, `entity_id`.
- Contact rollback path: `entity_type=contact` -> `fetch_contact` / `send_contact_custom_field_update`.
- Apply rollback still requires `ROLLBACK_DEAL_AWARE_AMO_FIELDS`.
- Real rollback against AMO was not executed.
