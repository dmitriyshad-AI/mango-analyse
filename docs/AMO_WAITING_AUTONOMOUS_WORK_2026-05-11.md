# AMO Waiting Autonomous Work, 2026-05-11

Scope: safe work that can be done while employees manually merge duplicate contacts in AMO/Tallanto.

No live AMO write is authorized by this stage.

## Quick Commands

Refresh the safe waiting-work pack and operator dashboard:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
make amo-waiting-pack
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_operator_status.py \
  --project-root . \
  --runtime-contract stable_runtime/CURRENT_RUNTIME.json \
  --out-root stable_runtime/operator_status_20260511_v4_waiting_work
```

Safe local checks that do not need AMO tunnel:

```bash
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_non_duplicate_quality_gate_command.sh
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_refresh_quality_gate_command.sh
```

## Runtime Folder

```text
stable_runtime/amo_waiting_autonomous_work_20260511_v1/
```

## Results

```text
text_quality_review_rows=3
text_quality_cleared_rows=3
non_duplicate_live_candidate_rows=1
contact_id_mismatch_rows=1
already_written_rows=53
refresh_candidate_rows=40
readback_missing_rows=15
```

Meaning:

- all 3 text-quality rows are now clean under the current CRM text detector;
- 1 not-yet-written row can go to a bounded quality-gate + AMO dry-run;
- 2 already-written text-quality rows move into refresh/readback logic, not live write;
- 1 contact-id mismatch remains blocked until operator verifies/repairs it;
- 40 already-written contacts differ from current payload and are refresh candidates;
- 15 already-written contacts need readback before refresh can be trusted.

## Green Local Gates

These were executed locally and passed:

```text
stable_runtime/amo_waiting_autonomous_work_20260511_v1/non_duplicate_quality_gate/summary.json
stable_runtime/amo_waiting_autonomous_work_20260511_v1/refresh_quality_gate/summary.json
```

## Blocked By Shared DB Tunnel

These real-tunnel dependent commands were attempted and stopped safely at preflight because `127.0.0.1:15432` refused the connection:

```text
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_non_duplicate_real_tunnel_dry_run_command.sh
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_refresh_real_tunnel_dry_run_command.sh
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_readback_missing_commands.sh
```

No CRM fields were written.

## Main Outputs

```text
non_duplicate_blockers_report.csv
text_quality_cleared_candidates_ru.csv
contact_id_mismatch_tasks.csv
already_written_refresh_diff.csv
already_written_refresh_candidates_ru.csv
readback_missing_written_rows.csv
dashboard.html
```

## After Employees Say "Готово"

Run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_waiting_autonomous_work_20260511_v1/run_post_merge_full_after_staff_done.sh
```

This command still does not live-write. It runs post-merge recheck, rebuilds candidates, runs quality/dry-run where possible, and refreshes operator status.

## If Shared DB Tunnel Is Available Earlier

Run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_readback_missing_commands.sh
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_non_duplicate_real_tunnel_dry_run_command.sh
stable_runtime/amo_waiting_autonomous_work_20260511_v1/next_refresh_real_tunnel_dry_run_command.sh
```

Then audit the generated reports before any live write.

PASS for these dry-runs means:

```text
live_write=false
preflight_failed=false
failed=0
expected_count_mismatch=false
dry_run count equals expected count
```

FAIL/BLOCK means: `preflight_failed=true`, `skipped>0`, `failed>0`, `live_write=true`, or input mismatch with the exact CSV used by the quality gate.

## Claude Audit

Audit pack:

```text
audits/_inbox/amo_waiting_autonomous_work_20260511_v1/
```

Command:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
claude -p --model opus --effort high --permission-mode acceptEdits \
  "/audit audits/_inbox/amo_waiting_autonomous_work_20260511_v1"
```
