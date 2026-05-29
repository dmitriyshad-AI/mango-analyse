# Backward compatibility

Legacy behavior:
- `_cross_brand_safe_template` was not changed.
- Legacy `FakeDraftProvider` path remains unchanged.
- Existing legacy cross-brand tests are still green as part of `tests/test_subscription_llm_draft_provider.py`.

v2 behavior:
- Adds a new v2-only template dispatcher after the existing content/safety verifier guards.
- If no template matches, v2 output remains unchanged.
- If a template changes text, the same `_reverify_dialogue_contract_text_change` hard-check is applied.

Compatibility result:
- No broad test regressions found in `test_subscription_llm_draft_provider.py` and `test_dialogue_contract_pipeline.py`.

