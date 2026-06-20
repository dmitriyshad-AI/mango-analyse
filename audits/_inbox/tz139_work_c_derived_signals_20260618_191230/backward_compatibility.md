# Backward Compatibility

- Existing DBs without `derived_signals.status` and `derived_signals.expires_at` migrate idempotently at `bootstrap`.
- Old signal rows are kept readable. Column `status` defaults to `active`; old `record_json` is not forcibly rewritten by migration.
- New DBs create `derived_signals` with lifecycle columns from the start.
- `stable_signal_id` formula was not changed; lifecycle updates do not create new signal IDs.
- Read API now hides resolved/stale/expired signals from profile, event children, and signal search. This is an intentional behavior change required by Work C.
- Existing non-managed signal types are not auto-closed by Work C recompute; only `paid_no_access`, `hot_lead_silent_7d`, and `duplicate_contact` are managed.
