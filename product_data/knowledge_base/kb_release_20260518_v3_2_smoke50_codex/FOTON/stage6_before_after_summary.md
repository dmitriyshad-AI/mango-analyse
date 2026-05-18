# Stage 6 KB enriched drafts summary

- created_at: 2026-05-18T18:50:57.645436+00:00
- input: product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_input/private_dialog_threads_FOTON.jsonl
- baseline_csv: product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_input/baseline_FOTON.csv
- snapshot: product_data/knowledge_base/kb_release_20260518_v3_2/kb_release_v3_snapshot.json
- snapshot_run_id: kb_release_20260518_v3_2
- out_dir: product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_codex/FOTON
- provider_mode: codex
- model: gpt-5.5
- reasoning_effort: xhigh

## Metrics

- rows_total: 25
- routes: {'draft_for_manager': 18, 'manager_only': 7}
- llm_or_provider_errors: 0
- invalid_topic_ids: 0
- unsupported_numeric_promises: 0
- brand_separation_violation: 0
- baseline_manager_only_relaxed: 0
- baseline_manager_only_preserved: 0
- used_kb_context: 25
- context_found_rate: 1.0
- became_more_substantive: 25
- empty_clarification_reduced: 25
- high_risk_route_relaxed: 0

## Safety

- live_telegram: false
- client_send: false
- write_crm: false
- write_tallanto: false
- write_stable_runtime: false
- run_asr: false
- run_resolve_analyze: false
