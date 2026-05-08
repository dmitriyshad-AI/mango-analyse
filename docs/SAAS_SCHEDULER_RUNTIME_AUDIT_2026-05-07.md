# SaaS scheduler runtime audit

Date: 2026-05-07

Scope: Stage 3 of the SaaS/productization plan: introduce a small supervisor /
scheduler runtime around the isolated product appliance DB.

## Result

Stage 3 is complete.

The product appliance DB now has a real `job_runs` runtime surface:

- scheduler migration `20260507_003_scheduler_runtime`
- planned `shadow_poll` jobs
- claim/lock execution
- retry/backoff for failed dry-run jobs
- stale lock requeue
- job output JSON under product appliance storage
- scheduler audit CLI
- tests for success, retries, lock behavior, disabled jobs, unsafe paths, and CLI

This is still intentionally a shadow/dry-run runtime. It does not download
audio and does not run ASR/R+A.

## Previous Stage Check

Stage 2 product DB admin remained healthy before this stage:

```text
validation_ok: true
blocked: 0
schema_migrations_before: 2
```

The only warnings are the known 3 pending CRM owner mappings from Stage 1:

```text
pending_owner_mappings: 3
```

Those warnings do not block Stage 3 because scheduler runtime does not assign
CRM owners or write CRM data.

## Safety boundary

This step did not:

- change `stable_runtime`
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- write to the runtime processing DB
- change current batch/start/run-ui scripts

Writes were limited to isolated productization code, tests, docs, and:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/
```

## New implementation

New module:

```text
src/mango_mvp/productization/scheduler_runtime.py
```

New CLI:

```text
scripts/mango_office_scheduler_runtime.py
```

Updated product DB module:

```text
src/mango_mvp/productization/product_db.py
```

Updated exports:

```text
src/mango_mvp/productization/__init__.py
```

New tests:

```text
tests/test_productization_scheduler_runtime.py
```

Updated tests:

```text
tests/test_productization_product_db.py
```

## Scheduler migration

Added migration:

```text
20260507_003_scheduler_runtime
```

New `job_runs` columns:

- `scheduled_for`
- `next_run_at`
- `attempt_count`
- `max_attempts`
- `lock_owner`
- `lock_expires_at`
- `heartbeat_at`
- `result_json`

New indexes:

```text
ix_job_runs_next_run
ix_job_runs_lock
```

Current migrations in the real product DB:

```text
20260507_001_product_appliance_base
20260507_002_config_history_retention
20260507_003_scheduler_runtime
```

## Enabled job type

Only this job type is executable in Stage 3:

```text
shadow_poll
```

The following job types remain disabled:

```text
capture_download
asr
ra
crm_sync
```

The scheduler blocks any disabled job if it appears in `job_runs`.

## Runtime guards

Every scheduled `shadow_poll` job carries hard guards:

```text
download_audio: false
run_asr: false
run_ra: false
write_crm: false
write_runtime_db: false
```

The executor validates those guards again from `input_ref` before running. It
also refuses raw payload and output paths outside the product appliance root.

## Real upgrade

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  upgrade \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_upgrade_stage3_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_upgrade_stage3_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
migrations_applied: 1
schema_migrations_before: 2
schema_migrations_after: 3
warnings: 3
```

## Raw payload copy

To keep the scheduler path guard strict, the existing shadow poll raw payload
archive was copied into the product appliance root:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl
```

The source artifact remains unchanged:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl
```

## Real scheduler pass

Plan command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  plan-shadow-poll \
  --tenant foton \
  --raw-payload _local_archive_mango_api_downloads_20260507/product_appliance/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl \
  --output-dir _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_plan_shadow_poll_stage3_final_audit.json
```

Tick command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  tick \
  --worker-id codex-stage3-worker \
  --limit 1 \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_tick_stage3_final_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs/shadow_poll_job_000003.json
```

Result:

```text
claimed: 1
succeeded: 1
failed: 0
blocked: 0
retry_wait: 0
validation_ok: true
raw_payload_rows: 665
```

Hard safety result:

```text
download_audio: false
run_asr: false
run_ra: false
write_crm: false
write_runtime_db: false
stable_runtime_writes: false
```

## Real scheduler audit

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  audit \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_runtime_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/scheduler_runtime_audit.json
```

Result:

```text
validation_ok: true
job_rows: 3
terminal_job_rows: 3
failed_job_rows: 0
blocked_job_rows: 0
retry_wait_job_rows: 0
running_job_rows: 0
status_counts:
  shadow_poll|succeeded: 3
warnings: 3
```

There are 3 succeeded jobs because the scheduler was re-run after hardening
changes so the final artifacts match the final implementation.

## Real product DB integrity

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_integrity_stage3_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
schema_migrations: 3
job_runs: 3
due_job_runs: 0
running_job_runs: 0
failed_job_runs: 0
warnings: 3
```

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_product_db.py \
  tests/test_productization_scheduler_runtime.py
```

Result:

```text
26 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Remaining limitations

This stage deliberately schedules a dry-run shadow job over an archived Mango
payload file inside the product appliance root. It does not yet run a long-lived
daemon and does not yet perform a live network poll on an interval.

The next stage can safely add one of these without touching the processing
runtime:

- cron/launchd wrapper for periodic `tick`
- provider credentials reference in tenant config
- live Mango read-only poll executor behind the same scheduler guards
- operator UI panel for `job_runs`
