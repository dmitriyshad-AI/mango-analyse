# Phase 12 TZ-05 olympiad_online · implementation notes

Status: formal_pass.

Scope:
- Registered `olympiad_online` in the v2 safe-template dispatcher with priority 50.
- Reused existing `_olympiad_online_safe_template` for explicit olympiad-online questions.
- Added a narrow v2 wrapper: if the current requested scope is `regular_online` and `olympiad_online` is blocked, the dispatcher returns a scope-safe handoff instead of confirming olympiad facts.
- Added topic normalization to `theme:016_program` when the template applies.
- Expanded `_known_grade_int` to parse "класса" as well as "класс/классе".

Constraints:
- TZ-08/TZ-09 not implemented.
- No KB changes, no LLM calls, no 212 run, no CRM/AMO/Tallanto writes, no live-send.

