# ADR-003 Source-Axis Blockers

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `d0946d65`
- Current handoff rows: `4`
- Route-only review candidates: `0`
- Source-axis blocked rows: `2`
- Alignment review unclear rows: `1`
- Danger-adjacent rows: `2`
- Shadow renderer candidates: `0`

## Next workstream

- `danger_adjacent_do_not_lower`: `2`
- `fix_proof_axis_alignment`: `2`

## Source alignment

- `blocked_source_axis_mismatch`: `2`
- `<empty>`: `1`
- `alignment_review_unclear`: `1`

## Cases

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` policy=`blocked_source_axis_mismatch`
  - next=`fix_proof_axis_alignment` source_alignment=`blocked_source_axis_mismatch` renderer=`blocked_source_axis_mismatch`
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, source_axis_mismatch`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` policy=`blocked_danger_adjacent`
  - next=`danger_adjacent_do_not_lower` source_alignment=`blocked_source_axis_mismatch` renderer=`blocked_source_axis_mismatch`
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, danger_adjacent_do_not_lower`
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`handoff_manager` policy=`blocked_danger_adjacent`
  - next=`danger_adjacent_do_not_lower` source_alignment=`` renderer=``
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, danger_adjacent_do_not_lower`
- `ra1_foton_platform_and_price#1` route=`manager_only` action=`answer_question` policy=`blocked_manager_only_route`
  - next=`fix_proof_axis_alignment` source_alignment=`alignment_review_unclear` renderer=`blocked_source_axis_review_unclear`
  - blockers: `report_only, no_text_generated, active_behavior_forbidden, manager_only_route`

## Acceptance Notes

- Active autonomy remains NO-GO: this report emits no route or text changes.
- Source/proof-axis blockers must clear before any text or route policy can be discussed.
- No route-only current handoff candidate is available in this queue.
