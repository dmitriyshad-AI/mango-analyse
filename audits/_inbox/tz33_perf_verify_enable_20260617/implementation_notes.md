# TZ-33 perf verify/enable — implementation notes

Date: 2026-06-17

Implemented only verification and default decisions for already existing flags.

Changes:

- `TALLANTO_BATCH_FETCH`: default `0 -> 1` in TZ-33, then `1 -> 0` in TZ-143 because `live_card_only` drops data used by compact contexts.
- `AMO_LEADS_BATCH_FETCH`: default `0 -> 1` in AMO deal resolution.
- `PROFILE_PHONE_INDEX`: default `0 -> 1` in profile store/summary lookup.
- Added/strengthened mock NEG tests for fewer requests and equivalent output.
- Added explicit default checks.
- Added TZ-143 invariant: default/OFF Tallanto keeps `opportunity_count>0` and `course_relation_count>0` through `compact_contexts` for both phone and contact-id paths.

No live AMO/Tallanto writes, no ASR, no Resolve+Analyze, no profile rebuild.
