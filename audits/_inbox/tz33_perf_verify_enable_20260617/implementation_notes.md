# TZ-33 perf verify/enable — implementation notes

Date: 2026-06-17

Implemented only verification and default enablement for already existing flags.

Changes:

- `TALLANTO_BATCH_FETCH`: default `0 -> 1` in Tallanto API/context.
- `AMO_LEADS_BATCH_FETCH`: default `0 -> 1` in AMO deal resolution.
- `PROFILE_PHONE_INDEX`: default `0 -> 1` in profile store/summary lookup.
- Added/strengthened mock NEG tests for fewer requests and equivalent output.
- Added explicit default-ON checks.

No live AMO/Tallanto writes, no ASR, no Resolve+Analyze, no profile rebuild.

