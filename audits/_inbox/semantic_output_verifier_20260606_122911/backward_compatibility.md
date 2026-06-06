# Backward Compatibility

Default behavior:
- `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` defaults OFF.
- Existing `TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD` path remains available when the new verifier is OFF.
- Existing deterministic gate codes keep their previous actions.

Data compatibility:
- `dynamic_summary.json` gains `semantic_output_verifier` and two `llm_calls` keys.
- Turn transcript gains `bot_semantic_output_verifier`.
- Existing consumers that ignore unknown JSON keys should continue to work.

Route compatibility:
- With verifier OFF, routes are unchanged.
- With verifier ON, only semantic findings can add `annotate` or `downgrade_keep_text`; final gate still cannot promote a route.
