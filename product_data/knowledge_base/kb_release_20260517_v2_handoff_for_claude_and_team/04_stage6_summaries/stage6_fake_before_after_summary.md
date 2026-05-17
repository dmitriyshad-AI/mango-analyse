# Stage 6 KB enriched drafts summary

- created_at: 2026-05-17T19:56:52.535029+00:00
- input: .codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/private_dialog_threads.jsonl
- baseline_csv: .codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_for_manual_review.csv
- snapshot: product_data/knowledge_base/kb_release_20260517_v2/kb_release_v2_snapshot.json
- snapshot_run_id: kb_release_20260517_v2
- out_dir: product_data/knowledge_base/kb_release_20260517_v2/stage6_fake_smoke
- provider_mode: fake
- model: fake
- reasoning_effort: fake

## Metrics

- rows_total: 20
- routes: {'draft_for_manager': 8, 'manager_only': 12}
- llm_or_provider_errors: 0
- invalid_topic_ids: 0
- unsupported_numeric_promises: 0
- baseline_manager_only_relaxed: 0
- baseline_manager_only_preserved: 11
- used_kb_context: 20
- context_found_rate: 1.0
- became_more_substantive: 11
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
