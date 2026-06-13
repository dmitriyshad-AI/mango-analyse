# TZ-19 calls review table implementation notes

Дата: 2026-06-13

## Implementation

- Added `scripts/build_tz19_calls_review_table.py`.
- Added `tests/test_tz19_calls_review_table.py`.
- Generated workbook outside git:
  `/Users/dmitrijfabarisov/Claude Projects/Foton/tz19_calls_review_table_2026-06-13.xlsx`
- Generated local summary outside git:
  `/Users/dmitrijfabarisov/Claude Projects/Foton/tz19_calls_review_table_2026-06-13.summary.json`

## Data source

- Canonical DB:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`
- Read-only SQLite URI with `mode=ro&immutable=1`.
- No ASR/Analyse/R+A rerun.

## Scope

The script defaults to `baseline_22679` because TZ-19 asks for the 22,679-row Analyse v7 set. Current DB has 26,118 v7 rows after TZ-21, so the script excludes the 3,439 TZ-21 tail manifest ids.

## Output shape

Sheets:

- `Все`
- `blacklist-77`
- `длинные`

One call is one row. The workbook contains ID, date, duration, transcript length, masked client id, AMO ids, brand best-effort, Analyse v7 columns, blacklist flag, and review flags.
