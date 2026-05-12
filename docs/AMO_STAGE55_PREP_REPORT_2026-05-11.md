# AMO Stage55 prep report 2026-05-11

## Scope

Prepared the next explicit AMO live-stage candidate pack after Claude `PASS_WITH_LIMITATIONS` on `amo_waiting_network_dryrun_20260511_v2`.

This report does **not** authorize live-write.

## Candidate payload

Root: `stable_runtime/amo_live_stage55_20260511_v1/`

Input CSV:

- `stable_runtime/amo_live_stage55_20260511_v1/live_stage55_candidates_ru.csv`

Source manifest:

- `stable_runtime/amo_live_stage55_20260511_v1/live_stage55_source_manifest.csv`

Counts:

- non-duplicate: `1`
- refresh: `40`
- repair from strict v5: `14`
- total: `55`

Excluded:

- `+79775501326`, reason: `multiple_exact_contacts_in_amo`, contact IDs `76169284 | 66349711`.

## Quality gate

Summary:

- `stable_runtime/amo_live_stage55_20260511_v1/stage55_quality_gate/summary.json`

Result:

- rows: `55`
- passed: `true`
- blocking_rows: `0`
- CRM text quality: `passed_for_live=true`
- frozen corpus: `64/64`, `0 failures`
- high precision population recall uncovered rows: `0`

Soft observations:

- C12 chronology overlap remains soft counter only.
- Review-precision marker `rv_learning_not_discussed` remains non-blocking by current policy.

## Dry-run status

Combined stage55 dry-run was attempted but blocked by AMO OAuth, not by data quality.

Failed dry-run:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260511T192253Z/contact_writeback_summary.json`
- result: `55 failed`
- reason: AMO OAuth returned `HTTP 401`, token expired/revoked.

Direct token fallback also failed:

- `stable_runtime/amocrm_runtime/contact_writebacks/20260511T192648Z/contact_writeback_summary.json`
- result: `1 failed`
- reason: `HTTP 401 Unauthorized`.

## OAuth blocker

Blocker file:

- `stable_runtime/amo_live_stage55_20260511_v1/stage55_oauth_blocker.json`

Reauthorization helper files:

- `stable_runtime/amo_live_stage55_20260511_v1/amocrm_reauthorize.html`
- `stable_runtime/amo_live_stage55_20260511_v1/AMO_REAUTHORIZE_URL.txt`
- `stable_runtime/amo_live_stage55_20260511_v1/refresh_amocrm_reauthorize_link.sh`
- `stable_runtime/amo_live_stage55_20260511_v1/amocrm_reauthorize_current.html`
- `stable_runtime/amo_live_stage55_20260511_v1/AMO_REAUTHORIZE_URL_CURRENT.txt`

Use the `*_CURRENT` direct URL, not the original amoCRM button, when amoCRM has rotated `client_id/state`.

To refresh and open the latest direct OAuth URL:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_live_stage55_20260511_v1/refresh_amocrm_reauthorize_link.sh
open "$(cat stable_runtime/amo_live_stage55_20260511_v1/AMO_REAUTHORIZE_URL_CURRENT.txt)"
```

After reauthorization, run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_live_stage55_20260511_v1/next_after_oauth_stage55_dry_run.sh
```

This command performs OAuth status check and then a dry-run only. It has no live-write flags.

## Next gate

After the post-OAuth stage55 dry-run returns `55/55 dry_run`, prepare a new Claude audit pack. Live-write can only be prepared after that audit passes.
