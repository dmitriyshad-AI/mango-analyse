# Stage 15 Export Quality Gate Report

Generated at: `2026-05-09T23:12:54.632791+00:00`
Gate version: `transcript_quality_stage15_export_gate_v2_hardened`

## Verdict

- Hard gate passed: `True`
- ROP/internal export ready: `True`
- CRM quality-writeback ready: `True`
- Bot allowlist export ready: `True`
- Autonomous bot production ready: `False`
- Autonomous bot blockers: `over_sanitization_queue_requires_rop_review_before_autonomous_bot`

## Hard Checks

- `required_files_exist`: `True`
- `stage14_acceptance_passed`: `True`
- `stage14_required_checks_passed`: `True`
- `stage14_residual_risk_rows_zero`: `True`
- `baseline_required_risks_zero`: `True`
- `stage14_inputs_match_current_roots`: `True`
- `audit_sample_sufficient_and_unique`: `True`
- `source_bot_safe_answers_have_zero_risks`: `True`
- `bot_export_allowlist_non_empty`: `True`
- `bot_export_allowlist_has_only_safe_columns`: `True`
- `bot_export_allowlist_has_zero_risks`: `True`

## Key Counts

- KB bot seed rows: `300`
- ROP bot draft rows: `250`
- Bot export allowlist rows: `473`
- Blocked bot export rows: `0`
- Stage 14 audit sample rows: `200`
- Stage 14 residual risk rows: `0`
- Over-sanitization queue rows: `250`

## Stage 14

- Acceptance flag: `True`
- Failed required checks: `none`
- Missing required checks: `none`

## Baseline Required Zero Risks

- `kb_no_live_revenue_risk`: `0`
- `kb_bot_ready_money_or_terms`: `0`
- `kb_ideal_answer_brand_risk`: `0`
- `kb_bot_safe_answer_brand_risk`: `0`
- `kb_bot_safe_answer_personal_data_risk`: `0`
- `rop_p0_no_live_or_artifact`: `0`
- `rop_revenue_risk_no_live_or_artifact`: `0`
- `rop_bot_candidate_money_or_terms`: `0`
- `rop_bot_safe_answer_brand_risk`: `0`
- `rop_bot_safe_answer_personal_data_risk`: `0`

## Output Policy

Use `bot_export_allowlist.csv` for bot/RAG ingestion. Do not ingest raw `bot_knowledge_drafts.csv`, `rop_validation.csv`, `enriched_reviews.csv`, or raw ideal/manager answer columns into an autonomous bot.

CRM writeback is quality-ready only if this gate passes. Live CRM writeback still requires the separate staged preview/live confirmation policy.

## Outputs

- `summary_json`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/summary.json`
- `export_gate_report_md`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/STAGE15_EXPORT_GATE_REPORT.md`
- `bot_export_allowlist_csv`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/bot_export_allowlist.csv`
- `bot_export_allowlist_schema_json`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/bot_export_allowlist.schema.json`
- `blocked_bot_export_rows_csv`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/blocked_bot_export_rows.csv`
- `export_gate_runbook_md`: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/transcript_quality_stage15_export_gate_20260510_v2_hardened/EXPORT_GATE_RUNBOOK.md`
