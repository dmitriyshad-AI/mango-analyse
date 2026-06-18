# 2026-06-18 Child resolver combined cache rerun

## Input

Re-ran the final child profile rebuild with a combined cache from:

- `/Users/dmitrijfabarisov/Projects/Mango_tz31_child_identity/product_data/customer_profiles/tz32_child_identity_full_v5_20260618_025225/llm_cache`
- `/Users/dmitrijfabarisov/Projects/Mango_tz31_child_identity/product_data/customer_profiles/tz34_child_escalation_v6_tier2_20260618_143739/llm_cache`

The merged cache contains `8086` files. This is the same count as the TZ34 cache, so TZ34 already contains TZ32 cache entries.

## Run

Output root:
`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/product_data/customer_profiles/prod_child_identity_dedup_20260618_final_combined_cache`

Command mode:

- `--cache-only`
- `--no-seed-tier1-cache`
- `--codex-cli-command /usr/bin/false`

No external LLM calls were possible or attempted.

Timeline snapshot:
`57843ec4239f4af400628778f4189b99885a6ace9dad247230415b6fb790ebf9`

Prompt/model versions unchanged:

- Tier 1: `gpt-5.4-mini`, prompt `v5`
- Tier 2: `gpt-5.5`, prompt `v6`

## Result

From `summary.json` and `after_escalation_build_report.json`:

- child slots: `11282`
- named slots: `5476`
- unnamed slots: `5806`
- profiles with 2+ children: `1378`
- raw child slots baseline: `17669`
- OFF resolver child slots baseline: `10217`
- `llm_calls_total`: `0`
- `llm_escalation_calls_total`: `0`
- `llm_cache_hits`: `7399`
- `llm_escalation_cache_hits`: `687`
- `llm_cache_misses_without_call`: `200`

## Cache Misses

The 200 cache misses remain even with the combined TZ32+TZ34 cache. That means those resolver decisions are absent from both cache folders, not merely missing from the copied output cache.

Miss-case exports:

- anonymized: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/product_data/customer_profiles/prod_child_identity_dedup_20260618_final_combined_cache/cache_miss_cache_only.anonymized.jsonl`
- raw local: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/product_data/customer_profiles/prod_child_identity_dedup_20260618_final_combined_cache/cache_miss_cache_only.raw.local.jsonl`

Sample anonymized case ids:

- `family_00f3697e8f5ade11`
- `family_02c9409d9e962323`
- `family_052a92302d6c4806`
- `family_0588def816889dde`
- `family_06a89a7d69f910ba`
- `family_07927b201cc5cecb`
- `family_09902f13b30630f3`
- `family_0a3e9fa79cba7d2e`
- `family_0b600b2404d685d9`
- `family_0fccf714056eedb5`

## Conclusion

Claude's concern about incomplete cache is mechanically valid: `llm_cache_misses_without_call=200` remains. However the accepted slot totals are already at the validated level (`11282`, not the OFF baseline `10217` and not raw `17669`). No new LLM run was made.
