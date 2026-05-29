# Backward compatibility

Legacy behavior:
- Existing `_olympiad_online_safe_template` is reused.
- Existing high-risk content guard path remains unchanged.

v2 behavior:
- Adds a dispatcher template at priority 50.
- Adds topic normalization metadata consistent with legacy behavior.
- Adds one parser improvement for "N класса".

Compatibility result:
- `tests/test_subscription_llm_draft_provider.py`: green.
- `tests/test_dialogue_contract_pipeline.py`: green.

