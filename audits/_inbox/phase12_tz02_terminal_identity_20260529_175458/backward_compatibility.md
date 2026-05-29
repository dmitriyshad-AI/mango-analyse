# Backward compatibility

Legacy behavior:
- `_terminal_safe_template` was not changed.
- Existing legacy tests in `tests/test_subscription_llm_draft_provider.py` remain green.

v2 behavior:
- Adds terminal template selection through the same single-winner dispatcher.
- Adds final identity safety-net before sanitize.
- Any text-changing identity guard is re-verified before returning.

Compatibility result:
- `tests/test_subscription_llm_draft_provider.py`: green.
- `tests/test_dialogue_contract_pipeline.py`: green.

