# Implementation Notes

Branch: `codex/d3-phase01-botsafe-integration`

Implemented:

- Cherry-picked D8 next-step extractor commit `7387998` as local commit `e268f40`.
- Added name scrubbing for `interest/title` only, preserving known program/org/brand names.
- Added retirement of stale bot-safe chunks that are no longer produced after brand re-resolution.

Changed code:

- `src/mango_mvp/customer_timeline/bot_safe_summary.py`
- `tests/test_customer_timeline_bot_safe_summary.py`

The production customer timeline DB was not opened for write. Final verification was performed on `/tmp/mango_phase01_integration_final_20260621_102959/customer_timeline.sqlite`.
