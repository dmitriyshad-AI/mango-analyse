# ADR-003 F2aa Action Gate Counterfactual Proof

Report-only counterfactual over the F2y current handoff fact-gap rows.

Changed files:

- `scripts/report_adr003_action_gate_counterfactual_proof.py`
- `tests/test_report_adr003_action_gate_counterfactual_proof.py`

Inputs:

- fact-gap report: `audits/_inbox/adr003_f2y_current_handoff_fact_gap_20260702170119/reports_36ea110/adr003_current_handoff_fact_gap_report.json`
- enriched transcripts: `audits/_inbox/adr003_f2v_existing_frame_proof_replay_20260702155709/reports_36ea110/enriched/dynamic_dialog_transcripts.jsonl`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Result:

- `cases=5`
- `scope_confusion_total=1`
- `action_only_still_blocked_total=4`
- `safe_reference_counterfactual_exact_proof_total=3`
- `counterfactual_residual_hard_missing_total=4`
- `negative_controls_preserved_total=2`
- `new_active_candidates=0`

Conclusion: changing only `requested_action` is insufficient; full safe-reference frame counterfactual can recover proof in some cases, but residual hard missing axes and policy blocks keep active autonomy at `NO-GO`.
