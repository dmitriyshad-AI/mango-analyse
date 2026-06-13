# TZ-21 risk review

## Data safety

- Raw profile DBs and comparison JSONs are under `product_data/customer_profiles/`, ignored by git.
- Stable runtime DB was changed only because TZ-21 explicitly required canonical import.
- Backup before write exists:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_20260613`
- A second backup was created for the idempotence apply run:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_idempotence_20260613`

## Write boundaries

- No AMO/CRM writes.
- No Tallanto writes.
- No email/Telegram sends.
- No ASR or Resolve+Analyze runs.
- No blacklist import.

## Main technical risk found and fixed

The profile rebuild initially failed because read-only SQLite URI opening did not handle the current project path and copied large DB state. The fix uses `mode=ro&immutable=1` and percent-escapes the path. Regression tests now cover paths with spaces.

## Residual risks

- `immutable=1` is safe for immutable/read-only inputs, but it must not be used for live DBs being modified concurrently. TZ-21 reads finished local files.
- Full profile content was not manually inspected in tracked form to avoid personal data leakage.
- The final working tree still contains unrelated untracked prompt/audit files from other threads; they were not modified or committed.
