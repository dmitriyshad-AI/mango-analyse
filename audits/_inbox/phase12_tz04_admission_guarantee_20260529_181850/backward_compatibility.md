# Backward compatibility

Legacy behavior:
- Existing legacy `ADMISSION_GUARANTEE_SAFE_TEXT` and trigger helper are reused.
- `_is_verified_safe_numeric_template` already includes `ADMISSION_GUARANTEE_SAFE_TEXT`.

v2 behavior:
- Adds a dispatcher template at priority 31, after result guarantee.
- Text-changing template output is still re-verified.
- Approved numeric safe templates are treated as their own grounding source during re-verification only.

Compatibility result:
- `tests/test_subscription_llm_draft_provider.py`: green.
- `tests/test_dialogue_contract_pipeline.py`: green.

