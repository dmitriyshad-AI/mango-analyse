# ADR-003 F2y Current Handoff Fact Gap

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `d9e957cd`
- Current handoff rows: `5`
- Non-danger rows: `3`
- Route-only candidates: `0`
- Proof axis mismatch: `1`
- Frame action blocks proof: `1`
- Partial facts but slot needed: `1`
- Danger excluded: `2`

## Root causes

- `danger_adjacent_do_not_lower`: `2`
- `frame_action_blocks_existence_proof`: `1`
- `partial_facts_available_but_slot_needed`: `1`
- `proof_axis_mismatch`: `1`

## Cases

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` next=`fix_proof_axis_alignment`
  - root: `proof_axis_mismatch`
  - proof: status=`would_reconcile_to_safe_reference` reason=`fresh_proof_contradicts_missing_facts_frame`
  - missing categories: `boarding_food, class_grade, dates_schedule, location_address, price_cost, program_direction`
  - fact categories: `boarding_food`
  - kb support: price=`18` platform=`0` product_check=`exists`
  - proven parts: `product_existence, price_cost`; missing slots: `grade`
  - next: Tighten proof/source alignment: a fact must cover every requested missing-fact axis before text readiness.
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` next=`fact_verification_or_retrieval_needed`
  - root: `frame_action_blocks_existence_proof`
  - proof: status=`blocked` reason=`no_exact_fact_keys`
  - missing categories: `boarding_food, class_grade, live_availability`
  - fact categories: ``
  - kb support: price=`0` platform=`0` product_check=`exists`
  - proven parts: `product_existence`; missing slots: ``
  - next: Calibrate SemanticFrame so stable existence/age suitability is answer_question, not check_availability.
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` next=`danger_adjacent_do_not_lower`
  - root: `danger_adjacent_do_not_lower`
  - proof: status=`blocked` reason=`no_exact_fact_keys`
  - missing categories: `class_grade, program_direction`
  - fact categories: ``
  - kb support: price=`0` platform=`0` product_check=`needs_slot`
  - proven parts: ``; missing slots: `grade`
  - next: Keep excluded from autonomy.
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`handoff_manager` next=`danger_adjacent_do_not_lower`
  - root: `danger_adjacent_do_not_lower`
  - proof: status=`blocked` reason=`requested_action_not_reconcilable`
  - missing categories: `class_grade, payment_access`
  - fact categories: ``
  - kb support: price=`0` platform=`0` product_check=`exists`
  - proven parts: `product_existence`; missing slots: ``
  - next: Keep excluded from autonomy.
- `ra1_foton_platform_and_price#1` route=`manager_only` action=`answer_question` next=`fact_verification_or_retrieval_needed`
  - root: `partial_facts_available_but_slot_needed`
  - proof: status=`blocked` reason=`no_exact_fact_keys`
  - missing categories: `class_grade, platform_current, price_cost`
  - fact categories: ``
  - kb support: price=`5` platform=`6` product_check=`needs_slot`
  - proven parts: `price_cost, platform_current`; missing slots: `grade`
  - next: Add partial-answer shadow: answer proven platform/format facts while asking only for the missing slot; report-only first.

## Acceptance Notes

- Active autonomy remains NO-GO: this report found no route-only candidate.
- Do not build a renderer before proof/source alignment covers requested missing-fact axes.
- manager_only rows remain manager_only until an owner-approved policy says otherwise.
- Partial-answer shadow may be worth measuring: answer proven facts while asking only for the missing slot.
