# Backward Compatibility

- Runtime code is not changed.
- All flags remain default-OFF.
- No profile/live configuration is changed.
- `bot_route` and `bot_text` are preserved in the enriched transcripts; replay summary reports `route_text_diff_count=0`.
- Existing SemanticFrame tests and calibration report tests pass with the new script test.
