# ADR-003 F2i SemanticFrame Calibration Queue

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `42ee2ff5`
- Gold compared rows: `77`
- Safe/self gold rows: `32`
- Manual too-cautious labels: `29`
- True frame must_handoff too-cautious: `10`
- True frame too-confident: `0`
- Current safe over-handoff candidates: `11`
- Strict active candidates now: `0`
- Manager-only exact-proof rows: `1`

## Real Lever Analysis

- True too-cautious rows: `10`
- Current handoff among too-cautious: `5`
- Fact assertion required: `8`
- Factless ack/status: `0`
- Danger-adjacent: `2`
- Clean route-only discussion rows: `0`
- Stable existence misread as check_availability: `0`
- Stable existence misread as enroll: `1`
- True live availability negative controls: `29`
- True enroll/booking negative controls: `9`

### Too-cautious by frame requested_action

- `answer_question`: `9`
- `enroll`: `1`

### Too-cautious classes

- `fact_assertion_required`: `8`
- `danger_adjacent_do_not_lower`: `2`

### Scope confusion

- total: `1`
- `enroll`: `1`

### Negative controls

- rows: `29`
- `wappi_pair_missing_72h_005#1` expected=`handoff_manager` frame_scope=`live_availability_or_enroll` frame_action=`handoff_manager`
- `wappi_pair_missing_72h_006#1` expected=`enroll` frame_scope=`live_availability_or_enroll` frame_action=`enroll`
- `wappi_pair_missing_72h_007#1` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `wappi_pair_missing_72h_009#1` expected=`check_availability` frame_scope=`manager_or_missing` frame_action=`answer_question`
- `wappi_pair_missing_72h_017#1` expected=`enroll` frame_scope=`live_availability_or_enroll` frame_action=`enroll`
- `wappi_pair_missing_72h_023#1` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `forward_payment_foton_hold_price_01#2` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `p0_model_led_pos_support_question#1` expected=`handoff_manager` frame_scope=`live_availability_or_enroll` frame_action=`handoff_manager`

### Real-lever examples

- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `manager_only_policy, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` class=`fact_assertion_required` confidence=`0.84`
  - blockers: `requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `wappi_pair_missing_72h_020#1` route=`bot_answer_self_for_pilot` action=`answer_question` class=`fact_assertion_required` confidence=`0.9`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` class=`danger_adjacent_do_not_lower` confidence=`0.86`
  - blockers: `danger_adjacent, frame_marks_operational_or_manager_risk`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`answer_question` class=`danger_adjacent_do_not_lower` confidence=`0.86`
  - blockers: `danger_adjacent, frame_marks_operational_or_manager_risk`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.
- `ra1_foton_address_and_trial#1` route=`bot_answer_self_for_pilot` action=`enroll` class=`fact_assertion_required` confidence=`0.88`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `ra1_foton_platform_and_price#1` route=`manager_only` action=`answer_question` class=`fact_assertion_required` confidence=`0.9`
  - blockers: `manager_only_policy, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_pos_unpk_camp_dates_signup#2` route=`bot_answer_self_for_pilot` action=`answer_question` class=`fact_assertion_required` confidence=`0.82`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_pos_unpk_exam_group_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_over_handoff_unpk_clean_ready#1` route=`bot_answer_self_for_pilot` action=`answer_question` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?

## Workstreams

- `semanticframe_existence_vs_availability`: `4`
- `semanticframe_safe_reference_missing_facts`: `9`
- `semanticframe_low_confidence`: `17`
- `retrieval_delivery_runtime_missing_exact_proof`: `1`
- `conversation_plan_scope_missing`: `1`
- `policy_manager_only_exact_proof`: `1`
- `policy_context_update_exact_proof`: `1`
- `danger_adjacent_do_not_lower`: `2`
- `already_self_no_active_leverage`: `21`
- `measurement_review_unclear`: `4`

## Priority Examples

### `semanticframe_existence_vs_availability`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` confidence=`0.84`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_pos_unpk_exam_group_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_over_handoff_unpk_clean_ready#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`

### `retrieval_delivery_runtime_missing_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `runtime_did_not_receive_exact_kb_proof`

### `conversation_plan_scope_missing`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `conversation_plan_no_product_scope`

### `policy_manager_only_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `manager_only_policy_blocks_even_with_fresh_exact_proof`

### `policy_context_update_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `context_update_policy_blocks_even_with_fresh_exact_proof`

### `semanticframe_low_confidence`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` confidence=`0.84`
  - reasons: `confidence_below:0.9`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `wappi_pair_missing_72h_004#1` route=`manager_only` action=`answer_question` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `rz_foton_warmth_trust_guarantee_09#2` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `ra1_foton_address_and_trial#1` route=`bot_answer_self_for_pilot` action=`enroll` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `ra1_unpk_unknown_slot_price#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.84`
  - reasons: `confidence_below:0.9`

## Acceptance Notes

- Active autonomy remains NO-GO: this is a calibration queue, not a behavior change.
- Existence/format vs live availability must be fixed in SemanticFrame/policy and validated in shadow first.
- manager_only/context_update rows need an explicit policy decision before any active demotion discussion.
- Fresh exact evidence alone is insufficient on current runtime telemetry.
- No draft_for_manager route-only active candidate is ready.
