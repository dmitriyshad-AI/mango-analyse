# AMO waiting network dry-run report 2026-05-11

## Result

Shared DB tunnel was brought up and the post-Claude AMO waiting network stage was executed without any live-write flags.

## Readback missing legacy rows

Source: `stable_runtime/amo_waiting_autonomous_work_20260511_v1/readback_missing_gate/summary.json`

- Evaluated rows: `15/15`
- Passed: `false`
- Blocking rows: `15`
- Risks:
  - `lossy_ellipsis_truncation`: `15`
  - `duplicate_label_and_count`: `10`

Interpretation: these are already-written legacy AMO rows with old bad CRM text. They are not authorized for live refresh as-is. They require repair from the current strict v5 source.

## Independent dry-runs

Non-duplicate dry-run:

- Report: `stable_runtime/amocrm_runtime/contact_writebacks/20260511T183739Z/contact_writeback_summary.json`
- Result: `1/1 dry_run`, `0 failed`

Refresh dry-run:

- Report: `stable_runtime/amocrm_runtime/contact_writebacks/20260511T184541Z/contact_writeback_summary.json`
- Result: `40/40 dry_run`, `0 failed`

Row 40 retry after tunnel drop:

- Report: `stable_runtime/amocrm_runtime/contact_writebacks/20260511T185620Z/contact_writeback_summary.json`
- Result: `1/1 dry_run`, `0 failed`

## Repair queue for 15 legacy-bad rows

Repair queue was rebuilt from the current strict v5 source:

- Input: `stable_runtime/amo_waiting_autonomous_work_20260511_v1/readback_blocked_repair_candidates_strict_v5_ru.csv`
- Quality gate: `stable_runtime/amo_waiting_autonomous_work_20260511_v1/readback_blocked_repair_strict_v5_quality_gate/summary.json`
- Quality result: `15/15 allow`, `passed=true`

Repair dry-run:

- Report: `stable_runtime/amocrm_runtime/contact_writebacks/20260511T185910Z/contact_writeback_summary.json`
- Result: `14 dry_run`, `1 skipped`, `0 failed`
- Skipped row: `+79775501326`, reason `multiple_exact_contacts_in_amo`, contact IDs `76169284 | 66349711`

Interpretation: 14 repair rows may be candidates for a future staged live pack after audit. The duplicate row must stay blocked until AMO duplicates are resolved manually.

## Code hardening done during this step

- Fixed readback risk aggregation so `lossy_ellipsis_truncation` is not split into broken fragments.
- Updated safe network command generation to run independent readback/dry-run steps and continue after an expected fail-closed step.
- Added per-step timeout support via `AMO_SAFE_NETWORK_STEP_TIMEOUT_SECONDS` to avoid hanging forever after a tunnel drop.
- Hardened SSH tunnel startup with `ConnectTimeout`, `TCPKeepAlive`, and faster keepalive detection.

## Tests

Targeted tests passed: `11 passed`.

## Audit pack

Prepared for Claude:

- `audits/_inbox/amo_waiting_network_dryrun_20260511_v2/`

This pack does not authorize live-write. It asks whether Codex may prepare the next explicit live-stage pack for:

- `1` non-duplicate row
- `40` refresh rows
- `14` repair rows

The duplicate repair row remains blocked.
