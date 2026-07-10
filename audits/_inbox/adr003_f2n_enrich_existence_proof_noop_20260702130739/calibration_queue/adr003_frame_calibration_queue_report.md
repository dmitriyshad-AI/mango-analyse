# ADR-003 F2i SemanticFrame Calibration Queue

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `6549b5b1`
- Gold compared rows: `77`
- Safe/self gold rows: `32`
- Manual too-cautious labels: `29`
- True frame must_handoff too-cautious: `14`
- True frame too-confident: `0`
- Current safe over-handoff candidates: `11`
- Strict active candidates now: `0`
- Manager-only exact-proof rows: `1`

## Workstreams

- `semanticframe_existence_vs_availability`: `9`
- `semanticframe_safe_reference_missing_facts`: `14`
- `semanticframe_low_confidence`: `16`
- `retrieval_delivery_runtime_missing_exact_proof`: `1`
- `conversation_plan_scope_missing`: `1`
- `policy_manager_only_exact_proof`: `1`
- `policy_context_update_exact_proof`: `1`
- `danger_adjacent_do_not_lower`: `2`
- `already_self_no_active_leverage`: `21`
- `measurement_review_unclear`: `3`

## Priority Examples

### `semanticframe_existence_vs_availability`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` confidence=`0.9`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `ra1_unpk_unknown_slot_price#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts`
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_pos_unpk_exam_group_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.88`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_p0_unpk_paid_transfer#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_fabrication_unpk_camp_medical#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_fabrication_unpk_camp_security#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.82`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts`
- `cf142_over_handoff_unpk_clean_ready#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`

### `retrieval_delivery_runtime_missing_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `runtime_did_not_receive_exact_kb_proof`

### `conversation_plan_scope_missing`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `conversation_plan_no_product_scope`

### `policy_manager_only_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `manager_only_policy_blocks_even_with_fresh_exact_proof`

### `policy_context_update_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `context_update_policy_blocks_even_with_fresh_exact_proof`

### `semanticframe_low_confidence`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78`
  - reasons: `confidence_below:0.9`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`answer_question` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `forward_payment_unpk_semester_year_01#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`handoff_manager` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `rz_foton_warmth_trust_guarantee_09#2` route=`manager_only` action=`answer_question` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `rz_foton_warmth_trust_guarantee_09#3` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `ra1_unpk_unknown_slot_price#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_pos_unpk_camp_dates_signup#2` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.82`
  - reasons: `confidence_below:0.9`

## Acceptance Notes

- Active autonomy remains NO-GO: this is a calibration queue, not a behavior change.
- Existence/format vs live availability must be fixed in SemanticFrame/policy and validated in shadow first.
- manager_only/context_update rows need an explicit policy decision before any active demotion discussion.
- Fresh exact evidence alone is insufficient on current runtime telemetry.
- No draft_for_manager route-only active candidate is ready.
