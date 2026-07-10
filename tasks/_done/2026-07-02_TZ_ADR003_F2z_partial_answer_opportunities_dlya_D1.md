# TZ ADR-003 F2z: Partial Answer Opportunities

Goal: scan the real 36ea110 enriched run for safe partial-answer opportunities after F2y showed `partial_facts_available_but_slot_needed`.

Scope:

- Add report-only script `scripts/report_adr003_partial_answer_opportunities.py`.
- Add regression tests `tests/test_report_adr003_partial_answer_opportunities.py`.
- Produce audit pack `audits/_inbox/adr003_f2z_partial_answer_opportunities_20260702171013/`.

Result:

- `total_turns=241`
- `handoff_turns=135`
- `partial_support_handoff_turns=44`
- `draft_partial_shadow_candidates=0`
- `manager_only_partial_policy_blocked=2`
- `hard_missing_axis_blocked=4`
- `broad_missing_axes_blocked=1`
- `action_or_danger_excluded_partial_rows=37`

Decision:

- F3 active autonomy remains `NO-GO`.
- No client text is generated.
- Next work should be counterfactual action/proof calibration, not a route switch.

Verification:

- F2z unit: 5 passed.
- ADR report tests: 48 passed.
- Live bot was not touched.
