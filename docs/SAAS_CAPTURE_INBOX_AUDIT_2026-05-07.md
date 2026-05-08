# SaaS capture inbox audit

Date: 2026-05-07

Scope: Stage 5 of the SaaS/productization plan: make live Mango
`ENQUEUE_SHADOW_CAPTURE` decisions durable without downloading audio and without
touching the current processing runtime.

## Result

Stage 5 is complete.

The product appliance DB now has a durable capture inbox:

- migration `20260507_004_capture_inbox`
- table `capture_inbox_items`
- idempotent upsert by `tenant_id + provider + event_key`
- source `job_runs` / report references
- raw Mango payload line references
- `ready_for_capture` status
- audit CLI and tests

This is not a download queue yet. It is the safe durable boundary between live
Mango polling and future recording capture.

## Previous Stage Check

Stage 4 live shadow poll was healthy before this stage:

```text
validation_ok: true
blocked_job_rows: 0
failed_job_rows: 0
running_job_rows: 0
retry_wait_job_rows: 0
status_counts:
  shadow_poll|succeeded: 5
```

Final live poll result from Stage 4:

```text
source_rows: 43
ENQUEUE_SHADOW_CAPTURE: 0
SKIP_DUPLICATE: 23
SKIP_NO_RECORDING: 20
validation_ok: true
```

The preceding live poll had 21 enqueue decisions. Stage 5 applied that report to
the inbox and then verified that the final repeated poll did not create
duplicates.

## Safety boundary

This step did not:

- change `stable_runtime`
- download audio
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- write to the runtime processing DB
- change current batch/start/run-ui scripts

Writes were limited to productization code/tests/docs and:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/
```

## New implementation

New module:

```text
src/mango_mvp/productization/capture_inbox.py
```

New CLI:

```text
scripts/mango_office_capture_inbox.py
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
tests/test_productization_capture_inbox.py
```

Updated tests:

```text
tests/test_productization_product_db.py
```

## Product DB migration

Added migration:

```text
20260507_004_capture_inbox
```

Current migrations:

```text
20260507_001_product_appliance_base
20260507_002_config_history_retention
20260507_003_scheduler_runtime
20260507_004_capture_inbox
```

Table:

```text
capture_inbox_items
```

Key fields:

- `tenant_id`
- `provider`
- `event_key`
- `provider_call_id`
- `status`
- `source_job_run_id`
- `source_report_ref`
- `raw_payload_ref`
- `started_at`
- `direction`
- `client_phone`
- `manager_ref`
- `recording_ref`
- `audio_ref`
- `candidate_json`
- `event_json`
- `first_seen_at`
- `last_seen_at`
- `enqueue_count`

Uniqueness:

```text
UNIQUE(tenant_id, provider, event_key)
```

## Real upgrade

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  upgrade \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_upgrade_stage5_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
migrations_applied: 1
schema_migrations_before: 3
schema_migrations_after: 4
warnings: 3
```

## Real apply

Applied Stage 4 live poll job with 21 enqueue decisions:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_capture_inbox.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  apply-report \
  --report _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs/shadow_poll_job_000004.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/capture_inbox_apply_stage5_from_job4_audit.json
```

Result:

```text
validation_ok: true
decisions_seen: 44
enqueue_decisions: 21
inserted: 21
updated_existing: 0
already_present: 0
skipped_non_enqueue: 23
inbox_items: 21
ready_for_capture: 21
blocked: 0
```

Re-applied the same report to verify idempotency:

```text
inserted: 0
updated_existing: 0
already_present: 21
inbox_items: 21
ready_for_capture: 21
validation_ok: true
```

Applied final Stage 4 live poll job with no enqueue decisions:

```text
decisions_seen: 43
enqueue_decisions: 0
inserted: 0
updated_existing: 0
skipped_non_enqueue: 43
inbox_items: 21
ready_for_capture: 21
validation_ok: true
```

## Real inbox audit

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_capture_inbox.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  audit \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/capture_inbox_stage5_audit.json
```

Result:

```text
validation_ok: true
items: 21
ready_for_capture: 21
missing_audio_ref: 0
duplicate_event_keys: 0
blocked: 0
warnings: 3
```

Manager refs in inbox:

| Manager ref | Items |
|---:|---:|
| 26 | 13 |
| 23 | 6 |
| 202 | 2 |

The 3 warnings are inherited from product DB integrity and remain the known
pending CRM owner mappings from Stage 1. They do not block capture inbox
durability.

## Product DB integrity

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_integrity_stage5_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
schema_migrations: 4
job_runs: 5
capture_inbox_items: 21
capture_inbox_ready: 21
capture_inbox_blocked: 0
product_calls: 297
warnings: 3
```

## Verification

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_product_db.py \
  tests/test_productization_capture_inbox.py \
  tests/test_productization_live_shadow_poll.py \
  tests/test_productization_scheduler_runtime.py
```

Result:

```text
37 passed, 1 warning
```

Full productization regression:

```text
124 passed, 1 warning
```

Insight-adjacent regression:

```text
32 passed
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Remaining limitations

The inbox intentionally stops before recording download. The next safe stage can
add a recording capture planner that reads `capture_inbox_items` and produces a
download dry-run manifest. Actual audio download should remain a separate
explicit stage after dry-run audit.
