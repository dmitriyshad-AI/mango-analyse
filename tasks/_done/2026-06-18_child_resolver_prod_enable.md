# 2026-06-18 Child resolver prod enable

## Scope

- Merged child identity resolver was already on `main`.
- Persisted production flags:
  - `PROFILE_LLM_CHILD_RESOLVER=1`
  - `PROFILE_LLM_CHILD_RESOLVER_ESCALATION=1`
  - `PROFILE_PHONE_INDEX=1`
- Final profile rebuild used fixed timeline snapshot:
  `57843ec4239f4af400628778f4189b99885a6ace9dad247230415b6fb790ebf9`.
- Reused existing `child_resolver_v1` cache from:
  `/Users/dmitrijfabarisov/Projects/Mango_tz31_child_identity/product_data/customer_profiles/tz34_child_escalation_v6_tier2_20260618_143739/llm_cache`.

No AMO/Tallanto writes, no ASR, no Resolve+Analyze.

## Build Output

Output root:
`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/product_data/customer_profiles/prod_child_identity_dedup_20260618_final`

Run config:
- `cache_only=true`
- Tier 1 model config preserved: `gpt-5.4-mini`, prompt `v5`
- Tier 2 escalation config preserved: `gpt-5.5`, prompt `v6`

## Verification

From `summary.json` / `after_escalation_build_report.json`:

- child slots: `11282`
- named slots: `5476`
- unnamed slots: `5806`
- raw slots baseline: `17669`
- OFF resolver slots baseline: `10217`
- `llm_calls_total`: `0`
- `llm_escalation_calls_total`: `0`
- `llm_cache_hits`: `7399`
- `llm_escalation_cache_hits`: `687`
- `llm_cache_misses_without_call`: `200`

The 200 cache misses were fail-soft under `cache_only=true`; no external LLM calls were attempted.

Phone index verification from `customer_profiles_after_escalation.sqlite`:

- `primary_phone_norm` column exists.
- `idx_customer_profiles_phone_norm` exists.
- filled `primary_phone_norm`: `18081 / 18399` profiles.

Persistent config verification:

- `.env.example`: all three flags are `1`.
- current worktree `.env`: all three flags are `1`.
- `/Users/dmitrijfabarisov/Projects/Mango analyse/.env`: all three flags are `1`.

## Tests

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`

Result: `3365 passed, 5 skipped, 1 warning`.

## Notes

The final rebuild added a cache-only mode to the batch runner so this acceptance run cannot silently burn new LLM calls when the expected cache is incomplete. Cache misses are counted separately and fail soft.
