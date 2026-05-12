# AMO Stage51 live preflight report 2026-05-12

## Scope

Prepared an explicit, guarded live-write stage after Claude Stage54 `PASS_WITH_LIMITATIONS`.

This report does **not** mean live-write has been executed.

## Why Stage51

Claude Stage54 allowed preparing a live-stage script, but flagged 3 review-precision rows as `needs_review`:

- `+79165261123`
- `+79161492492`
- `+79859618552`

Instead of writing all 54 rows, Stage51 excludes these 3 rows and writes only rows with Claude decision `allow`.

Already excluded before Stage51:

- `+79272761437`: `wrong_person_or_identity_mismatch`
- `+79775501326`: duplicate AMO contact / multiple exact contacts

## Candidate Payload

Root:

- `stable_runtime/amo_live_stage51_20260512_v1/`

Live candidate CSV:

- `stable_runtime/amo_live_stage51_20260512_v1/live_stage51_candidates_ru.csv`

Needs-review CSV:

- `stable_runtime/amo_live_stage51_20260512_v1/stage51_needs_review_ru.csv`

Counts:

- live candidates: `51`
- needs_review excluded: `3`
- wrong-person excluded from Stage54: `1`
- duplicate-AMO excluded earlier: `1`

## Quality Gate

Summary:

- `stable_runtime/amo_live_stage51_20260512_v1/stage51_quality_gate/summary.json`

Result:

- rows: `51`
- passed: `true`
- blocking_rows: `0`
- tenant_config.loaded: `true`
- tenant_id: `foton`
- tenant_config sha256: `9de1e6363171ea619cdd52055ce16f0b2b71c499a6d00fd88b2f55e70711c288`
- frozen corpus: `69/69`, `0 failures`
- population recall: `0` high-precision uncovered, `0` review uncovered

## Dry-Run

Real-tunnel dry-run:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260511T224711Z/contact_writeback_summary.json`

Result:

- `51/51 dry_run`
- `0 skipped`
- `0 failed`
- `live_write=false`

## Live Script

Approval helper:

- `stable_runtime/amo_live_stage51_20260512_v1/approve_stage51_live_write.sh`

Live + readback script:

- `stable_runtime/amo_live_stage51_20260512_v1/next_live_stage51_then_readback.sh`

The live script is fail-closed. It requires:

- approval file: `stable_runtime/amo_live_stage51_20260512_v1/operator_approval_stage51.json`
- env confirmation: `CONFIRM_AMO_STAGE51_LIVE_WRITE=WRITE_AMO_LIVE_STAGE51_20260512`
- `--execute-live-write`
- `--live-confirmation WRITE_AMO_LIVE`
- `--expected-written 51`
- immediate post-writeback readback gate with `--expected-evaluated 51`

## Claude Preflight Pack

Audit pack:

- `audits/_inbox/amo_stage51_live_script_preflight_20260512_v1/`

Run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
claude -p --model opus --effort high --permission-mode acceptEdits \
  "/audit audits/_inbox/amo_stage51_live_script_preflight_20260512_v1"
```

## Current Recommendation

Run Claude preflight first. If it returns PASS or PASS_WITH_LIMITATIONS explicitly allowing manual execution, Дмитрий may execute:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_live_stage51_20260512_v1/approve_stage51_live_write.sh
CONFIRM_AMO_STAGE51_LIVE_WRITE=WRITE_AMO_LIVE_STAGE51_20260512 \
  stable_runtime/amo_live_stage51_20260512_v1/next_live_stage51_then_readback.sh
```

Do not run Stage54/Stage55 live scripts. Stage51 is the current guarded candidate.
