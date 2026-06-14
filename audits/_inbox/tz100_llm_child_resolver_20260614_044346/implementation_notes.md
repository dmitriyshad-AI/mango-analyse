# TZ100 LLM child resolver implementation notes

Date: 2026-06-14

Scope:
- Added flag-gated `PROFILE_LLM_CHILD_RESOLVER` path, default OFF.
- OFF path keeps the previous deterministic `apply_child_slot_merge_candidates` behavior.
- ON path runs before `apply_superseded_rules`, inside the existing child-slot merge step.
- Added `child_resolver_v1` LLM response cache through existing `LLMResponseCache`.
- Added bounded batch execution with `ThreadPoolExecutor`, patterned after the question catalog LLM assigner.
- Added mandatory shared-phone stoplist loading from `~/.mango_secrets/shared_phones_stoplist.json`.
- Added fail-soft family-level handling: bad model output/veto leaves that family unchanged.
- Added anonymized microprobe trace under ignored `product_data/customer_profiles/`.

Main implementation:
- `src/mango_mvp/customer_profile/child_resolver_llm.py`
- `src/mango_mvp/customer_profile/builder.py`
- `scripts/run_tz100_child_resolver_microprobe.py`
- `tests/test_customer_profile_builder.py`

Microprobe:
- Output: `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz100_microprobe_20260614_043317/`
- Selected families: 18
- Shared-phone candidate included: true
- LLM provider/model: `codex_cli` / `gpt-5.4-mini`
- LLM cases: 18
- Calls after cache on final run: 7
- Cache hits on final run: 10
- Shared-phone skipped: 1
- Families resolved: 10
- Families fail-soft/vetoed: 7
- Child slots before -> after on microprobe: 90 -> 85
- Profiles with 2+ children before -> after: 16 -> 10

No AMO/CRM/Tallanto writes were executed.
No ASR or Resolve+Analyze was executed.
The full 7,512-family run was not executed.
