# ADR-003 F2g Manager-Only Exact-Proof Root Cause

- Status: `pass_diagnosed`
- Source rev: `322020d`
- Manager-only exact-proof rows: `2`
- Runtime exact proof missing: `2`
- Conversation plan lacks product scope: `2`
- Frame says manager action: `1`
- Low confidence: `1`

## Root Cause Codes

- `answer_contract_no_required_fact_keys`: `2`
- `conversation_plan_no_product_scope`: `2`
- `frame_action_not_safe_reference`: `1`
- `frame_confidence_below_threshold`: `1`
- `frame_marks_manager_action`: `1`
- `route_locked_manager_only`: `2`
- `runtime_missing_facts_present`: `2`
- `runtime_retrieval_missed_exact_fact`: `2`
- `runtime_retrieval_zero_candidates`: `2`
- `self_shadow_has_no_runtime_exact_fact_keys`: `2`

## Cases

- `wappi_pair_missing_72h_001#1` route=`manager_only` frame=`safe/answer_self` action=`answer_question` confidence=`0.86`
  - exact fact: `lvsh_mendeleevo_2026.directions.fizmat.classes` status=`exists`
  - runtime retrieval: candidate_count=`0`, selected_exact_ids=`[]`, selected_adjacent_ids=`[]`
  - plan: primary_intent=`general_consultation`, topic_id=`service:S5_general_consultation`, product_scope=``, required_fact_keys=`[]`
  - root causes: `route_locked_manager_only, runtime_retrieval_missed_exact_fact, runtime_retrieval_zero_candidates, conversation_plan_no_product_scope, answer_contract_no_required_fact_keys, frame_confidence_below_threshold, runtime_missing_facts_present, self_shadow_has_no_runtime_exact_fact_keys`
- `wappi_pair_missing_72h_003#1` route=`manager_only` frame=`manager_action/manager_only` action=`check_availability` confidence=`0.92`
  - exact fact: `lvsh_mendeleevo_2026.directions.fizmat.classes` status=`exists`
  - runtime retrieval: candidate_count=`0`, selected_exact_ids=`[]`, selected_adjacent_ids=`[]`
  - plan: primary_intent=`general_consultation`, topic_id=`service:S5_general_consultation`, product_scope=``, required_fact_keys=`[]`
  - root causes: `route_locked_manager_only, runtime_retrieval_missed_exact_fact, runtime_retrieval_zero_candidates, conversation_plan_no_product_scope, answer_contract_no_required_fact_keys, frame_marks_manager_action, frame_action_not_safe_reference, runtime_missing_facts_present, self_shadow_has_no_runtime_exact_fact_keys`

## Acceptance Notes

- Active F3 remains NO-GO: this report diagnoses manager_only rows only.
- If runtime retrieval missed exact proof, the next safe step is shadow evidence injection or retrieval diagnostics, not manager_only demotion.
- If frame marks manager_action, frame calibration is required before any active route change.
- Report-only: no route/text/runtime changes.
