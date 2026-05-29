# Phase 12 TZ-03 result_guarantee · implementation notes

Status: formal_pass.

Scope:
- Registered `result_guarantee` in the v2 safe-template dispatcher with priority 30.
- Reused `RESULT_GUARANTEE_INPUT_RE` / `_is_result_guarantee_case`.
- Added the missing "точно сдаст на 90+" promise-context branch to the existing result guarantee regex.
- Kept `result_guarantee` as a safe rejection template that can replace an unsupported numeric promise draft.
- Added re-verification support for approved numeric safe templates by grounding their own text as `_verified_safe_numeric_template`.

Constraints:
- TZ-04/05/06/07 not included in this commit.
- TZ-08/TZ-09 not implemented.
- No KB changes, no LLM calls, no 212 run, no CRM/AMO/Tallanto writes, no live-send.

