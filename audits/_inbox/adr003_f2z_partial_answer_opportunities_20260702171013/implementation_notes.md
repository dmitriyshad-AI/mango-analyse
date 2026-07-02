# ADR-003 F2z Partial Answer Opportunities

Report-only scan over the real 36ea110 enriched run.

Changed files:

- `scripts/report_adr003_partial_answer_opportunities.py`
- `tests/test_report_adr003_partial_answer_opportunities.py`

Inputs:

- enriched transcripts: `audits/_inbox/adr003_f2v_existing_frame_proof_replay_20260702155709/reports_36ea110/enriched/dynamic_dialog_transcripts.jsonl`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Result:

- `total_turns=241`
- `handoff_turns=135`
- `partial_support_handoff_turns=44`
- `draft_partial_shadow_candidates=0`
- `manager_only_partial_policy_blocked=2`
- `hard_missing_axis_blocked=4`
- `broad_missing_axes_blocked=1`
- `action_or_danger_excluded_partial_rows=37`

Conclusion: partial KB support exists, but there are no clean draft-route partial-answer candidates on this run. Active autonomy remains `NO-GO`.
