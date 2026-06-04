# Backward Compatibility

## Flag OFF

`TELEGRAM_Q_THREAD_MEMORY` defaults to OFF. Regression test `test_q_thread_memory_flag_off_does_not_promote_known_slots_without_topic_focus` verifies that known-slots-only memory does not change deterministic retrieval when the flag is absent.

## Accepted Baseline

Existing `topic_focus` restoration remains unchanged. This was already present in the accepted base and is not newly gated by Block 2.

## Existing Safety Paths

- P0 remains manager-only.
- Explicit subject switch does not inherit old subject.
- Camp-family does not fall back to regular-course fact.
- Active brand remains channel-provided.
- Product price estimates without facts remain blocked with quality flags ON.

## Test Status

Full pytest with the accepted v6.4 artifact:

```text
2506 passed, 5 skipped, 1 warning in 42.64s
```
