# Operator runtime status, 2026-05-11

This document records the new read-only operator layer that sits above the post-backfill call-processing and AMO writeback artifacts.

## What was added

- `stable_runtime/CURRENT_RUNTIME.json` is now the machine-readable runtime contract.
- `scripts/mango_office_current_runtime.py` rebuilds that contract from the current active artifacts.
- `scripts/mango_office_operator_status.py` builds the operator status pack.
- `stable_runtime/operator_status_20260511_v1/operator_status.json` is the current status endpoint payload.
- `stable_runtime/operator_status_20260511_v1/operator_dashboard.html` is the first static operator dashboard.
- `stable_runtime/operator_status_20260511_v1/crm_queue_operator.csv` is the human-readable CRM queue with Russian action explanations.

## Current status

- Runtime contract: green.
- Call processing: green.
- Canonical actionable calls: `64 832`.
- Missing ASR: `0`.
- Missing Resolve+Analyze: `0`.
- AMO-ready strict rows: `69`.
- Already written/readback-verified or previously written: `53`.
- Ready rows for immediate next live stage: `0`.
- Manual-resolution rows: `16`.

## Verification

- Focused regression for manual-resolution/workbook/pipeline/operator/Product API/writeback guards: `63 passed`.
- Product API focused regression: `21 passed`.
- Full project regression: `811 passed`, `1` non-blocking `urllib3/LibreSSL` warning.
- Runtime contract command: passed.
- Operator status command: passed.
- Product API readiness: passed, `16` read-only endpoints.
- Product API HTTP readiness: passed, `16` read-only routes.

## AMO production-loop state

Current stage: `stage1_complete_no_ready_rows`.

Live writeback is intentionally blocked now because there are no remaining rows in `ready_single_contact_not_written`.

Next production actions are not mass writeback actions. They are resolution actions:

1. Resolve `12` multi-contact rows by selecting the correct AMO contact or merging duplicates.
2. Resolve `1` contact-id mismatch row by comparing dry-run contact ID vs source AMO contact IDs.
3. Review `3` text-quality rows, then rebuild the strict export and gates.

Only after those rows move into a fresh ready bucket should the writeback sequence resume:

`strict_export -> crm_quality_gate -> queue_classification -> real_tunnel_dry_run -> operator_live_confirmation -> staged_live_write -> post_writeback_readback -> queue_rebuild`

## Commands

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
make runtime-contract
make runtime-status
```

Direct commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_current_runtime.py --out stable_runtime/CURRENT_RUNTIME.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_operator_status.py --out-root stable_runtime/operator_status_20260511_v1
```

## Safety

Both commands are read-only with respect to CRM, Tallanto, audio, ASR and R+A. They only write local status/report/dashboard artifacts.

They do not:

- download audio;
- run ASR;
- run Resolve+Analyze;
- write to AMO;
- write to Tallanto;
- delete or move cleanup quarantine files.

## Manual-resolution workflow

The manual-resolution workflow for the remaining `16` AMO queue rows is now built:

- script: `scripts/build_amo_manual_resolution_pack.py`;
- implementation: `src/mango_mvp/productization/amo_manual_resolution.py`;
- current pack: `stable_runtime/amo_manual_resolution_20260511_v1/`;
- audit pack: `audits/_inbox/amo_manual_resolution_operator_status_20260511_v1/`.

Current pack result:

- Review rows: `16`.
- Accepted rows: `0`.
- Resolved live candidates: `0`.
- Needs human: `14`.
- Already-written review: `2`.
- Live writeback executed: `false`.
- `summary.blocked` in operator status means runtime/production-loop blocking reasons, while contact-id mismatch rows are exposed separately as `contact_id_mismatch_blocked_rows`.

The workflow is fail-closed: unresolved rows cannot enter live writeback. Only rows with an accepted resolution status and a validated contact id can enter `resolved_live_candidates_ru.csv`.

Accepted rows also require `resolution_reason` and `resolved_by`. Text-quality rows require a `text_quality_approved` reason; already-written refreshes require a `refresh_approved` reason; any contact id outside the source AMO set requires `allow_contact_id_outside_source=yes` and an `outside_source_approved` reason.

Accepted statuses:

`accepted`, `accepted_auto_policy`, `accepted_by_manager`, `accepted_by_operator`

After accepted decisions are provided, the required sequence remains:

`apply decisions -> build resolved candidates -> CRM quality gate -> real-tunnel dry-run -> explicit operator live approval artifact -> staged live write -> post-writeback readback -> queue rebuild`

## Next recommended development step

Resolve the `14` rows in `needs_human.csv` and decide what to do with the `2` already-written review rows. Then rebuild the manual-resolution pack with a filled decisions CSV. If the resolved-candidate CSV becomes non-empty, run the next dry-run command from `stable_runtime/amo_manual_resolution_20260511_v1/next_dry_run_command.sh`.


## 2026-05-11 Duplicate Merge Update

The 12 multi-contact AMO rows are now treated as duplicate-merge blockers, not ordinary manual accepted rows. They cannot enter live writeback until AMO duplicates are merged manually and post-merge dry-run recheck is green. Any accepted decision for `needs_manager_review_multi_contact` must include `post_merge_recheck_approved` in `resolution_reason`.

Current duplicate-resolution pack:

```text
stable_runtime/amo_duplicate_resolution_20260511_v1/
```
