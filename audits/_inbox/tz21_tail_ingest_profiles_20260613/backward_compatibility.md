# TZ-21 backward compatibility

## Code compatibility

- `CustomerProfileBuilder` public options and return shape are unchanged.
- `scripts/build_tz16_profiles_v7.py` CLI arguments are unchanged.
- `scripts/compute_tz16_rerun_tail.py` CLI arguments are unchanged.
- The SQLite opening change only affects read-only source connections.

## Data compatibility

- Existing `tz16_profiles_v7_20260612` output was not modified.
- New TZ-21 output is a separate ignored folder:
  `product_data/customer_profiles/tz21_profiles_after_tail_20260613/`
- Source `tz12_working_batch3` hash before/after is identical.
- Canonical call DB kept valid JSON count stable at 65,939 and only moved 3,439 rows from old summaries to v7 summaries.

## Known naming debt

The reusable profile build script still has `tz16` in its name/schema/build id. This is backward-compatible and intentionally left unchanged in TZ-21 to avoid broad renaming.
