# ADR-003 F2y Current Handoff Fact Gap

Report-only diagnostic for the current handoff rows found by F2x on the M1 36ea110 run.

Changed files:

- `scripts/report_adr003_current_handoff_fact_gap.py`
- `tests/test_report_adr003_current_handoff_fact_gap.py`

Runtime behavior is unchanged. The script only reads:

- F2x queue report: `audits/_inbox/adr003_f2x_current_handoff_proof_alignment_20260702164552/reports_36ea110/adr003_frame_calibration_queue_report.json`
- enriched transcripts: `audits/_inbox/adr003_f2v_existing_frame_proof_replay_20260702155709/reports_36ea110/enriched/dynamic_dialog_transcripts.jsonl`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Result:

- `current_handoff_rows=5`
- `route_only_candidates=0`
- `danger_excluded=2`
- `proof_axis_mismatch=1`
- `frame_action_blocks_proof=1`
- `partial_facts_slot_needed=1`

Conclusion: F3 route-only activation remains `NO-GO`. The next useful work is calibration and fact-proof work, not a behavior switch.
