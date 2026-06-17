# TZ-143 Tallanto batch default OFF

Date: 2026-06-18

Branch/worktree: `codex/tz33-perf-verify-enable` in `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`.

## Change

Reverted only `TALLANTO_BATCH_FETCH` default from `"1"` to `"0"` in the two requested code points:

- `src/mango_mvp/amocrm_runtime/tallanto_api.py`
- `src/mango_mvp/amocrm_runtime/tallanto_context.py`

Kept ON:

- `AMO_LEADS_BATCH_FETCH` in `src/mango_mvp/amocrm_runtime/deals.py`
- `PROFILE_PHONE_INDEX` in both `src/mango_mvp/customer_profile/store.py` and `src/mango_mvp/customer_profile/crm_summary.py`

## Reason

`TALLANTO_BATCH_FETCH=1` enables `live_card_only`, which hard-drops `opportunities`, `requests`, and `course_relations`. These fields feed `compact_contexts` (`opportunity_count`, `course_relation_count`) and are therefore not a neutral performance optimization for deal-aware card consumers.

## Test invariant

Added a consumer-level invariant through `build_live_tallanto_context`:

- default/OFF phone path calls `build_contact_context(..., live_card_only=False)`;
- default/OFF contact-id path calls `build_contact_context_by_contact_id(..., live_card_only=False)`;
- both preserve `opportunity_count>0` and `course_relation_count>0` in `contexts[0]`.

## Tests

Tallanto:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tallanto_api.py
```

Result: `11 passed in 0.05s`.

Full:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Result: `3329 passed, 5 skipped, 1 warning in 52.08s`.

Warning: system Python urllib3/OpenSSL warning, unrelated.

No AMO/Tallanto live calls, no ASR, no Resolve+Analyze, no profile rebuild.

