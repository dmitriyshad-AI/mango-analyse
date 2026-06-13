# TZ-19 calls review table backward compatibility

## Code compatibility

- New script is additive: `scripts/build_tz19_calls_review_table.py`.
- No existing runtime code paths were changed.
- No DB schema changes.
- No AMO/Tallanto/CRM code paths touched.

## Data compatibility

- Canonical DB was read only.
- Output workbook is regenerated deterministically at the same default path for the same date, so repeat runs do not create duplicate files.
- `--scope current_v7` is available if a later reviewer wants all current 26,118 v7 rows, but the default remains the 22,679-row baseline requested by TZ-19.
