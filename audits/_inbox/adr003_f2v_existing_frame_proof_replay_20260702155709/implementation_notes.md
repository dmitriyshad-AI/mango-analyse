# Implementation Notes

## Scope

Implemented an offline replay harness for ADR-003 proof shadows:

- `scripts/enrich_adr003_existing_frame_proof_shadow.py`
- `tests/test_enrich_adr003_existing_frame_proof_shadow.py`

The script enriches saved M1 transcripts with proof-shadow telemetry by reusing already stored `bot_semantic_frame` values. It does not recompute SemanticFrame and does not call an LLM.

## Raw Run

Input transcripts:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_clean_36ea110_20260702/runs/adr003_f2_self_answer_shadow_36ea110/ON/dynamic_dialog_transcripts.jsonl`

KB snapshot:

`product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Outputs:

- `reports_36ea110/enriched/dynamic_dialog_transcripts.jsonl`
- `reports_36ea110/enriched/existing_frame_proof_shadow_summary.json`
- `reports_36ea110/enriched/existing_frame_proof_shadow_summary.md`
- `reports_36ea110/calibration_queue/adr003_frame_calibration_queue_report.json`
- `reports_36ea110/calibration_queue/adr003_frame_calibration_queue_report.md`

## Key Numbers

- `turns_total=241`
- `turns_with_frame=241`
- `model_calls_added=0`
- `route_text_diff_count=0`
- `proof_reconciliation_would_reconcile_to_safe_reference=9`
- `proof_reconciliation_send_as_is_review_candidates=0`
- `fact_gated_strict_f3_draft_candidates=0`

## Interpretation

The replay confirms there is a proof-reconciliation signal, but it is not yet an active route-only lever. The current active candidate count remains zero because safe self-answering still needs a text policy/template layer and semantic review.
