# ADR-003 F2i SemanticFrame Calibration Queue

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `cf1497ff`
- Gold compared rows: `77`
- Safe/self gold rows: `32`
- Manual too-cautious labels: `29`
- True frame must_handoff too-cautious: `14`
- True frame too-confident: `0`
- Current safe over-handoff candidates: `11`
- Strict active candidates now: `0`
- Fact-gated strict draft candidates: `0`
- Fact-gated manager-only exact-proof rows: `2`
- Fact-gated already-self exact-proof rows: `6`
- Fact-gated blocked no-proof rows: `1`
- Manager-only exact-proof rows: `2`
- Proof reconciliation would-fix-frame rows: `9`

## Real Lever Analysis

- True too-cautious rows: `14`
- Current handoff among too-cautious: `5`
- Fact assertion required: `12`
- Factless ack/status: `0`
- Danger-adjacent: `5`
- Clean route-only discussion rows: `0`
- Proof reconciliation would-fix-frame rows: `2`
- Proof reconciliation send-as-is review candidates: `0`
- Proof reconciliation text blocked: `2`
- Proof source fact text ready: `1`
- Proof source fact text present: `2`
- Proof source fact PII signal: `0`
- Proof template registry found: `0`
- Proof direct quote forbidden: `2`
- Stable existence misread as check_availability: `7`
- Stable existence misread as enroll: `1`
- True live availability negative controls: `29`
- True enroll/booking negative controls: `9`

### Too-cautious by frame requested_action

- `check_availability`: `7`
- `answer_question`: `4`
- `handoff_manager`: `2`
- `enroll`: `1`

### Too-cautious classes

- `fact_assertion_required`: `9`
- `danger_adjacent_do_not_lower`: `5`

### Scope confusion

- total: `9`
- `check_availability`: `7`
- `enroll`: `1`
- `handoff_manager`: `1`

### Negative controls

- rows: `29`
- `wappi_pair_missing_72h_005#1` expected=`handoff_manager` frame_scope=`live_availability_or_enroll` frame_action=`handoff_manager`
- `wappi_pair_missing_72h_006#1` expected=`enroll` frame_scope=`live_availability_or_enroll` frame_action=`enroll`
- `wappi_pair_missing_72h_007#1` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `wappi_pair_missing_72h_009#1` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `wappi_pair_missing_72h_017#1` expected=`enroll` frame_scope=`live_availability_or_enroll` frame_action=`enroll`
- `wappi_pair_missing_72h_023#1` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `forward_payment_foton_hold_price_01#2` expected=`check_availability` frame_scope=`live_availability_or_enroll` frame_action=`check_availability`
- `p0_model_led_pos_support_question#1` expected=`handoff_manager` frame_scope=`live_availability_or_enroll` frame_action=`handoff_manager`

### Real-lever examples

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` class=`fact_assertion_required` confidence=`0.92`
  - blockers: `manager_only_policy, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `wappi_pair_missing_72h_020#1` route=`bot_answer_self_for_pilot` action=`handoff_manager` class=`fact_assertion_required` confidence=`0.93`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` class=`danger_adjacent_do_not_lower` confidence=`0.86`
  - blockers: `danger_adjacent, frame_marks_operational_or_manager_risk`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`handoff_manager` class=`danger_adjacent_do_not_lower` confidence=`0.82`
  - blockers: `danger_adjacent, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.
- `ra1_foton_address_and_trial#1` route=`bot_answer_self_for_pilot` action=`enroll` class=`fact_assertion_required` confidence=`0.93`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `ra1_foton_platform_and_price#1` route=`manager_only` action=`answer_question` class=`fact_assertion_required` confidence=`0.88`
  - blockers: `manager_only_policy, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`check_availability` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_pos_unpk_camp_dates_signup#2` route=`bot_answer_self_for_pilot` action=`answer_question` class=`fact_assertion_required` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_pos_unpk_exam_group_signup#1` route=`bot_answer_self_for_pilot` action=`check_availability` class=`fact_assertion_required` confidence=`0.92`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: What exact fresh client-safe fact would justify a self-answer?
- `cf142_p0_unpk_paid_transfer#1` route=`bot_answer_self_for_pilot` action=`check_availability` class=`danger_adjacent_do_not_lower` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, danger_adjacent, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.
- `cf142_fabrication_unpk_camp_medical#1` route=`bot_answer_self_for_pilot` action=`check_availability` class=`danger_adjacent_do_not_lower` confidence=`0.86`
  - blockers: `already_self_or_no_route_leverage, requires_verified_fact_assertion, danger_adjacent, frame_marks_operational_or_manager_risk, requested_action_not_safe_reference`
  - review: Keep out of active candidates because the dialog is close to P0/money/fabrication.

## Fact-Gated Readiness Summary

- strict draft candidates: `0`
- manager-only exact-proof rows: `2`
- already-self exact-proof rows: `6`
- blocked no exact proof: `1`
- excluded danger/money/P0: `1`

## Proof Reconciliation Shadow

- traced rows: `241`
- would reconcile: `9`
- already aligned: `35`
- `requested_action_not_reconcilable`: `74`
- `no_exact_fact_keys`: `60`
- `protected_p0`: `53`
- `frame_already_safe_answer_self`: `35`
- `fresh_proof_contradicts_missing_facts_frame`: `9`
- `deal_stage_blocked`: `7`
- `payment_readiness_blocked`: `3`

## Proof Reconciliation Text Readiness

- traced would-reconcile rows: `9`
- send-as-is review candidates: `0`
- active behavior allowed: `False`
- source fact client_safe_text present: `9`
- source fact client_safe_text PII signal: `0`
- bot_template_required: `1`
- template registry found: `1`
- direct quote forbidden: `9`
- structured_value available: `9`

### Text readiness by status

- `blocked_no_route_leverage_currently_self_or_other`: `8`
- `blocked_deferral_or_manager_text_signal`: `1`

### Text readiness blockers

- `missing_facts_present`: `9`
- `semantic_verifier_unavailable`: `9`
- `no_route_leverage_currently_self_or_other`: `8`
- `answer_quality_findings_present`: `3`
- `deferral_or_manager_text_signal`: `1`

### Source fact lookup

- `wrong_brand`: `5`
- `found`: `4`

### Source text candidate readiness

- `blocked_wrong_brand`: `5`
- `source_text_ready`: `3`
- `blocked_bot_template_required`: `1`

### Text policy readiness

- `blocked_wrong_brand`: `5`
- `source_text_ready_requires_nonquote_policy`: `3`
- `template_registry_found_requires_renderer`: `1`

### Template registry

- `not_required`: `8`
- `found`: `1`

## Workstreams

- `semanticframe_existence_vs_availability`: `7`
- `semanticframe_safe_reference_missing_facts`: `6`
- `semanticframe_proof_reconciliation_candidate`: `2`
- `semanticframe_low_confidence`: `13`
- `retrieval_delivery_runtime_missing_exact_proof`: `2`
- `conversation_plan_scope_missing`: `2`
- `policy_manager_only_exact_proof`: `2`
- `policy_context_update_exact_proof`: `2`
- `danger_adjacent_do_not_lower`: `2`
- `already_self_no_active_leverage`: `21`
- `measurement_review_unclear`: `4`

## Priority Examples

### `semanticframe_existence_vs_availability`
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:manager_action, frame_answerability:manager_only`
- `cf142_pos_unpk_exam_group_signup#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.92`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:missing_facts, frame_answerability:manager_only`
- `cf142_p0_unpk_paid_transfer#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:manager_action, frame_answerability:manager_only`
- `cf142_fabrication_unpk_camp_medical#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.86`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:manager_action, frame_answerability:manager_only`
- `cf142_fabrication_unpk_camp_security#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.88`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:manager_action, frame_answerability:manager_only`
- `cf142_over_handoff_unpk_clean_ready#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.94`
  - reasons: `safe_existence_or_format_labeled_as_manager_or_operational, frame_action:check_availability, frame_risk:manager_action, frame_answerability:manager_only`

### `retrieval_delivery_runtime_missing_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `runtime_did_not_receive_exact_kb_proof`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` confidence=`0.92`
  - reasons: `runtime_did_not_receive_exact_kb_proof`

### `conversation_plan_scope_missing`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `conversation_plan_no_product_scope`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` confidence=`0.92`
  - reasons: `conversation_plan_no_product_scope`

### `policy_manager_only_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `manager_only_policy_blocks_even_with_fresh_exact_proof`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` confidence=`0.92`
  - reasons: `manager_only_policy_blocks_even_with_fresh_exact_proof`

### `policy_context_update_exact_proof`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `context_update_policy_blocks_even_with_fresh_exact_proof`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` confidence=`0.92`
  - reasons: `context_update_policy_blocks_even_with_fresh_exact_proof`

### `semanticframe_low_confidence`
- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` action=`handoff_manager` confidence=`0.82`
  - reasons: `confidence_below:0.9`
- `rz_foton_warmth_trust_guarantee_09#2` route=`manager_only` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `ra1_foton_platform_and_price#1` route=`manager_only` action=`answer_question` confidence=`0.88`
  - reasons: `confidence_below:0.9`
- `ra1_unpk_unknown_slot_price#1` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_pos_unpk_camp_dates_signup#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_pos_unpk_camp_dates_signup#2` route=`bot_answer_self_for_pilot` action=`answer_question` confidence=`0.86`
  - reasons: `confidence_below:0.9`
- `cf142_p0_unpk_paid_transfer#1` route=`bot_answer_self_for_pilot` action=`check_availability` confidence=`0.86`
  - reasons: `confidence_below:0.9`

## Acceptance Notes

- Active autonomy remains NO-GO: this is a calibration queue, not a behavior change.
- Existence/format vs live availability must be fixed in SemanticFrame/policy and validated in shadow first.
- manager_only/context_update rows need an explicit policy decision before any active demotion discussion.
- Fresh exact evidence alone is insufficient on current runtime telemetry.
- No draft_for_manager route-only active candidate is ready.
