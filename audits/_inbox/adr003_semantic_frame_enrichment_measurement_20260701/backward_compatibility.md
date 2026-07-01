# Backward Compatibility

- Default runtime behavior is unchanged: all SemanticFrame flags remain default OFF.
- `--semantic-frame-enrich-from` is an opt-in simulator/reporting mode.
- The enrichment path does not call the draft provider's main generation method and does not mutate route/text/safety/checklist.
- Existing `--transcripts-in` and `--replay-from` remain mutually exclusive with the new enrichment input.
- Parallel enrichment changes only measurement runtime. Output order is preserved from the input transcript.
- Same-payload `TELEGRAM_SEMANTIC_FRAME_SHADOW` remains available for investigation, but is no longer the canonical no-op measurement path.
