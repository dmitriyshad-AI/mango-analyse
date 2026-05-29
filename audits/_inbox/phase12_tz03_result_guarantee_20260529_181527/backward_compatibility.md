# Backward compatibility

Legacy behavior:
- Existing legacy result-guarantee template is reused.
- `_is_verified_safe_numeric_template` already contained `RESULT_GUARANTEE_SAFE_TEXT`; no legacy whitelist change needed.

v2 behavior:
- Adds a dispatcher template at priority 30.
- Text-changing template output is still re-verified.
- `result_guarantee_safe_template_applied` and `placeholder_in_draft` are preserved for downstream compatibility.

Compatibility result:
- `tests/test_subscription_llm_draft_provider.py`: green.
- `tests/test_dialogue_contract_pipeline.py`: green.

