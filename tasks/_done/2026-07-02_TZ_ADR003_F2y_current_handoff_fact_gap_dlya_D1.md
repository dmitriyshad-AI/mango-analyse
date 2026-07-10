# TZ ADR-003 F2y: Current Handoff Fact Gap

Goal: explain the current handoff rows from the real 36ea110 F2/F2x artifacts and determine whether any route-only autonomy candidate exists.

Scope:

- Add report-only script `scripts/report_adr003_current_handoff_fact_gap.py`.
- Add regression tests `tests/test_report_adr003_current_handoff_fact_gap.py`.
- Produce audit pack `audits/_inbox/adr003_f2y_current_handoff_fact_gap_20260702170119/`.

Result:

- `current_handoff_rows=5`
- `route_only_candidates=0`
- `danger_excluded=2`
- `proof_axis_mismatch=1`
- `frame_action_blocks_proof=1`
- `partial_facts_slot_needed=1`

Decision:

- F3 active autonomy remains `NO-GO`.
- Next work should be report-only calibration/fact-proof work, not a route switch.

Verification:

- F2y unit: 6 passed.
- ADR report tests: 43 passed.
- Full pytest: 3912 passed, 5 skipped, 1 warning.
- Live bot was not touched.
