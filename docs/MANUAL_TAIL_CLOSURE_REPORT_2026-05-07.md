# Manual R+A Tail Closure Report - 2026-05-07

## Outcome

Closed the final `468` calls that had ASR but were stuck in `resolve_status=manual` / `analysis_status=pending`.

Final strict coverage report:

- Source audio in date window 2025-01-01..2026-05-31: `64,867`
- Excluded manager-manager no-ASR calls: `35`
- Actionable source audio: `64,832`
- ASR done: `64,832 / 64,832`
- Full R+A done: `64,832 / 64,832`
- Missing ASR: `0`
- Missing full R+A: `0`
- Manual not full R+A: `0`
- Coverage errors: `[]`

Report files:

- `stable_runtime/final_processing_coverage_report_20260507_v5/summary.json`
- `stable_runtime/final_processing_coverage_report_20260507_v5/coverage_by_month.tsv`
- `stable_runtime/final_processing_coverage_report_20260507_v5/missing_asr.txt`
- `stable_runtime/final_processing_coverage_report_20260507_v5/missing_full_ra.txt`
- `stable_runtime/final_processing_coverage_report_20260507_v5/manual_not_full_ra.txt`

## Method

I did not mutate old historical batch DBs. Instead I created a separate fallback DB:

- `stable_runtime/manual_tail_analyze_fallback_20260507/manual_tail_analyze_fallback_20260507.db`

The DB contains only the 468 manual-tail rows copied from:

- `stable_runtime/ra_missing_all_20260506/ra_missing_all_20260506.db`

For copied rows:

- `transcription_status = done`
- `resolve_status = skipped`
- `analysis_status = pending`, then processed to `done`
- `resolve_json` documents the fallback decision: analyze raw `transcript_text` because Resolve could not confidently accept a speaker merge

This makes the records terminal for strict coverage while preserving the fact that speaker resolution was not confidently accepted.

## Analyze Result Distribution

Final `quality_flags.call_type` distribution in the fallback DB:

- `non_conversation`: `239`
- `technical_call`: `116`
- `service_call`: `84`
- `sales_call`: `27`
- `existing_client_progress`: `2`

This confirms the tail was not just empty noise: a meaningful subset contained useful CRM/sales context and received real Analyze output.

## Supporting Artifacts

Manual-tail audit:

- `stable_runtime/manual_tail_audit_20260507/summary.json`
- `stable_runtime/manual_tail_audit_20260507/manual_tail_best.tsv`
- `stable_runtime/manual_tail_audit_20260507/samples_by_bucket.md`

Fallback preparation script:

- `scripts/prepare_manual_tail_analyze_fallback.py`

Worker launcher used for this run:

- `stable_runtime/manual_tail_analyze_fallback_20260507/start-analyze-6.sh`

## Operational Note

`stable_runtime/venv_stable/bin/python` currently lacks `sqlalchemy`, so this run used system `python3` with `PYTHONPATH=src`. Also, `codex exec` needed a writable `CODEX_HOME` under `/private/tmp/mango_codex_home` because the sandbox cannot write to `~/.codex/sessions`.
