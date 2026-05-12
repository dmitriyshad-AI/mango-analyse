# AMO Stage54 prep report 2026-05-11

## Scope

Prepared corrected AMO writeback stage after post-OAuth Stage55 dry-run.

This report does **not** authorize live-write.

## Why Stage55 Was Superseded

Stage55 dry-run was mechanically successful:

- `55/55 dry_run`
- `0 skipped`
- `0 failed`
- `live_write=false`

Local payload audit then found one true CRM relevance blocker:

- phone: `+79272761437`
- AMO contact ID: `76059894`
- class: `wrong_person_or_identity_mismatch`
- evidence: contact not confirmed, wrong-name/person context, discussion of program/product/next steps did not happen.

This was fixed as a general class, not a literal phone-only exclusion.

## General Fix

Updated detector/counter/corpus:

- `src/mango_mvp/quality/crm_writeback_quality_detector.py`
- `src/mango_mvp/quality/crm_writeback_population_recall.py`
- `tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl`
- `tests/test_crm_writeback_quality_detector.py`
- `tests/test_crm_writeback_population_recall.py`
- `docs/THREAT_MODEL.md`

New class:

- `wrong_person_or_identity_mismatch`

Negative overblock guard:

- valid EdTech objection `это не та программа` must remain allowed when student/product/course context exists.

## Candidate Payload

Root:

- `stable_runtime/amo_live_stage54_20260511_v1/`

Input CSV:

- `stable_runtime/amo_live_stage54_20260511_v1/live_stage54_candidates_ru.csv`

Source manifest:

- `stable_runtime/amo_live_stage54_20260511_v1/live_stage54_source_manifest.csv`

Excluded row:

- `stable_runtime/amo_live_stage54_20260511_v1/stage54_excluded_by_quality_gate.csv`

Counts:

- non-duplicate: `1`
- refresh: `39`
- repair: `14`
- total: `54`
- excluded by v6 relevance gate: `1`

## Quality Gate

Summary:

- `stable_runtime/amo_live_stage54_20260511_v1/stage54_quality_gate/summary.json`

Result:

- rows: `54`
- passed: `true`
- blocking_rows: `0`
- frozen corpus: `66/66`, `0 failures`
- high precision population recall uncovered rows: `0`

## Dry-Run

Real-tunnel dry-run:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260511T203615Z/contact_writeback_summary.json`

Result:

- `54/54 dry_run`
- `0 skipped`
- `0 failed`
- `live_write=false`

## Audit Pack

Claude audit pack:

- `audits/_inbox/amo_stage54_post_oauth_dryrun_20260511_v1/`

Run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_live_stage54_20260511_v1/next_claude_audit_command.sh
```

## Next Gate

If Claude returns PASS or PASS_WITH_LIMITATIONS that explicitly permits preparing live-stage, Codex may prepare a separate live-write script for these same 54 rows.

Live-write execution still requires explicit human action and post-writeback readback gate.
