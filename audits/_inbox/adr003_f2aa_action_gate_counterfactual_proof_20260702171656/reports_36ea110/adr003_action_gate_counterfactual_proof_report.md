# ADR-003 F2aa Action Gate Counterfactual Proof

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `d6013c24`
- Cases: `5`
- Scope confusion cases: `1`
- Action-only still blocked: `4`
- Safe-reference exact proof: `3`
- Residual hard missing after proof: `4`
- New active candidates: `0`

## Statuses

- `negative_control_preserved`: `2`
- `action_only_exact_proof_but_residual_hard_missing`: `1`
- `safe_reference_exact_proof_but_residual_hard_missing`: `1`
- `safe_reference_no_exact_proof`: `1`

## Cases

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` root=`proof_axis_mismatch` status=`action_only_exact_proof_but_residual_hard_missing`
  - current: action=`answer_question` risk=`missing_facts` answerability=`manager_only` must_handoff=`True`
  - action-only: `exists`/`exact_product_existence_fact`
  - safe-reference: `exists`/`exact_product_existence_fact`
  - residual: `boarding_food, dates_schedule, location_address, price_cost`; why not active: `report_only, active_behavior_allowed_false, residual_missing_categories_present`
- `wappi_pair_missing_72h_003#1` route=`manager_only` root=`frame_action_blocks_existence_proof` status=`safe_reference_exact_proof_but_residual_hard_missing`
  - current: action=`check_availability` risk=`manager_action` answerability=`manager_only` must_handoff=`True`
  - action-only: `blocked`/`protected_handoff_frame`
  - safe-reference: `exists`/`exact_product_existence_fact`
  - residual: `boarding_food, live_availability`; why not active: `report_only, active_behavior_allowed_false, route_manager_only, residual_hard_missing_axes, residual_missing_categories_present`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` root=`danger_adjacent_do_not_lower` status=`negative_control_preserved`
  - current: action=`answer_question` risk=`missing_facts` answerability=`manager_only` must_handoff=`True`
  - action-only: `blocked`/`required_axis_missing`
  - safe-reference: `blocked`/`required_axis_missing`
  - residual: `class_grade, program_direction`; why not active: `report_only, active_behavior_allowed_false, negative_control, residual_missing_categories_present`
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` root=`danger_adjacent_do_not_lower` status=`negative_control_preserved`
  - current: action=`handoff_manager` risk=`manager_action` answerability=`manager_only` must_handoff=`True`
  - action-only: `blocked`/`protected_handoff_frame`
  - safe-reference: `exists`/`exact_product_existence_fact`
  - residual: `payment_access`; why not active: `report_only, active_behavior_allowed_false, negative_control, residual_missing_categories_present`
- `ra1_foton_platform_and_price#1` route=`manager_only` root=`partial_facts_available_but_slot_needed` status=`safe_reference_no_exact_proof`
  - current: action=`answer_question` risk=`missing_facts` answerability=`manager_only` must_handoff=`True`
  - action-only: `blocked`/`required_axis_missing`
  - safe-reference: `blocked`/`required_axis_missing`
  - residual: `class_grade, platform_current, price_cost`; why not active: `report_only, active_behavior_allowed_false, route_manager_only, residual_missing_categories_present`

## Acceptance Notes

- Active autonomy remains NO-GO: counterfactuals do not change runtime behavior.
- Action-only calibration is insufficient when must_handoff/risk_class remain manager_action.
- Safe-reference exact proof still requires residual-missing-axis review and owner policy.
