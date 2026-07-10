# TZ ADR-003 F2aa: Action Gate Counterfactual Proof

Goal: verify whether `requested_action` alone explains the `frame_action_blocks_existence_proof` class, and whether a full safe-reference frame would recover exact proof.

Scope:

- Add report-only script `scripts/report_adr003_action_gate_counterfactual_proof.py`.
- Add regression tests `tests/test_report_adr003_action_gate_counterfactual_proof.py`.
- Produce audit pack `audits/_inbox/adr003_f2aa_action_gate_counterfactual_proof_20260702171656/`.

Result:

- `cases=5`
- `scope_confusion_total=1`
- `action_only_still_blocked_total=4`
- `safe_reference_counterfactual_exact_proof_total=3`
- `counterfactual_residual_hard_missing_total=4`
- `negative_controls_preserved_total=2`
- `new_active_candidates=0`

Decision:

- F3 active autonomy remains `NO-GO`.
- Do not patch only `requested_action`.
- Next work, if pursued, is full frame tuple calibration and residual proof-policy work.

Verification:

- F2aa unit: 5 passed.
- Live bot was not touched.
