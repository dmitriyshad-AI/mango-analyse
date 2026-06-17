# Backward compatibility

- Env vars remain the compatibility control:
  - `TALLANTO_BATCH_FETCH=1` for the optimized Tallanto path; default is OFF after TZ-143.
  - `AMO_LEADS_BATCH_FETCH=0`
  - `PROFILE_PHONE_INDEX=0`
- No public function signatures changed.
- No live AMO/Tallanto writes were added.
- Existing databases without `primary_phone_norm` still work because summary lookup checks for the column and falls back when it is absent.
- `CustomerProfileSQLiteStore` creates/fills `primary_phone_norm` only on profile write/rebuild when the flag is enabled.
