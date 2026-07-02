# ADR-003 F2ac Partial Answer Policy Shadow

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `8cd2d966`
- Partial draft candidates in input: `2`
- Joined with calibration queue: `2`
- Policy shadow candidates: `0`
- Blocked by danger adjacency: `1`
- Blocked by source-axis mismatch: `1`

## Status breakdown

- `blocked_danger_adjacent`: `1`
- `blocked_source_axis_mismatch`: `1`

## Cases

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` partial=`draft_partial_shadow_candidate` policy=`blocked_source_axis_mismatch`
  - queue: workstream=`fix_proof_axis_alignment` danger=`False` source_alignment=`blocked_source_axis_mismatch`
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, source_axis_mismatch`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` partial=`draft_partial_shadow_candidate` policy=`blocked_danger_adjacent`
  - queue: workstream=`danger_adjacent_do_not_lower` danger=`True` source_alignment=`blocked_source_axis_mismatch`
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, danger_adjacent_do_not_lower`

## Acceptance Notes

- Active autonomy remains NO-GO: this report emits no route or text changes.
- Partial-answer candidates must survive cross-report queue blockers before any text policy discussion.
- No partial-answer policy candidate survives cross-report blockers.
