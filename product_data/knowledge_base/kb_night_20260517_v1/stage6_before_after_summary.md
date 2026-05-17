# Stage 6 KB enriched drafts summary

- created_at: 2026-05-17T00:53:45.033359+00:00
- input: .codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/private_dialog_threads.jsonl
- baseline_csv: .codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_for_manual_review.csv
- snapshot: product_data/knowledge_base/kb_night_20260517_v1/kc_snapshot_20260517_night_v1.json
- snapshot_run_id: 20260517_night_v1
- out_dir: product_data/knowledge_base/kb_night_20260517_v1
- provider_mode: fake
- model: fake
- reasoning_effort: fake

## Metrics

- rows_total: 20
- routes: {'draft_for_manager': 15, 'manager_only': 5}
- llm_or_provider_errors: 0
- invalid_topic_ids: 0
- unsupported_numeric_promises: 0
- used_kb_context: 20
- context_found_rate: 1.0
- became_more_substantive: 19
- empty_clarification_reduced: 1
- high_risk_route_relaxed: 0

## Safety

- live_telegram: false
- client_send: false
- write_crm: false
- write_tallanto: false
- write_stable_runtime: false
- run_asr: false
- run_resolve_analyze: false
