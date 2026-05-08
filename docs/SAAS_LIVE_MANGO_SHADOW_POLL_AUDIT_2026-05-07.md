# SaaS live Mango shadow poll audit

Date: 2026-05-07

Scope: Stage 4 of the SaaS/productization plan: connect real Mango Office
read-only polling to the product appliance scheduler.

## Result

Stage 4 is complete.

The scheduler can now plan and execute a live Mango read-only shadow poll:

- credentials are loaded from environment variables only;
- secrets are not stored in `job_runs.input_ref`;
- Mango `stats/request` and `stats/result` are called through `MangoOfficeClient`;
- raw provider rows are archived under the product appliance root;
- rows are normalized into `TelephonyCallEvent`;
- decisions are produced by `CapturePlanner`;
- the product DB is used only as a seen-key/audit source;
- no audio is downloaded;
- ASR/R+A/CRM/runtime writes remain disabled.

## Previous Stage Check

Stage 3 scheduler runtime was healthy before this stage:

```text
validation_ok: true
blocked_job_rows: 0
failed_job_rows: 0
running_job_rows: 0
retry_wait_job_rows: 0
status_counts:
  shadow_poll|succeeded: 3
```

Product DB integrity was also healthy:

```text
validation_ok: true
blocked: 0
schema_migrations: 3
```

The only remaining warnings are the known 3 pending CRM owner mappings from
Stage 1.

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
src/mango_mvp/productization/mango_live_shadow_poll.py
```

Updated scheduler runtime:

```text
src/mango_mvp/productization/scheduler_runtime.py
```

Updated scheduler CLI:

```text
scripts/mango_office_scheduler_runtime.py
```

Updated exports:

```text
src/mango_mvp/productization/__init__.py
```

New tests:

```text
tests/test_productization_live_shadow_poll.py
```

Updated tests:

```text
tests/test_productization_scheduler_runtime.py
```

## Scheduler command

New command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  plan-live-shadow-poll \
  --tenant foton \
  --window-hours 2 \
  --raw-payload-dir _local_archive_mango_api_downloads_20260507/product_appliance/raw_payload_archive/live_shadow_poll \
  --output-dir _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_plan_live_shadow_poll_stage4_final_audit.json
```

The scheduled job input stores only credential references:

```text
api_key: env:MANGO_OFFICE_API_KEY
api_salt: env:MANGO_OFFICE_API_SALT
```

Hard guards:

```text
download_audio: false
run_asr: false
run_ra: false
write_crm: false
write_runtime_db: false
```

## Real live poll

Final tick command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_scheduler_runtime.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  tick \
  --worker-id codex-stage4-live-worker \
  --limit 1 \
  --lock-seconds 600 \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/scheduler_tick_live_shadow_poll_stage4_final_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/scheduler_outputs/shadow_poll_job_000005.json
```

Raw payload archive:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/raw_payload_archive/live_shadow_poll/live_shadow_poll_foton_20260507T140331Z_20260507T160331Z.jsonl
```

Final live poll result:

```text
source_rows: 43
raw_payload_rows: 43
normalized_events: 43
normalization_errors: 0
seen_event_keys: 318
ENQUEUE_SHADOW_CAPTURE: 0
SKIP_DUPLICATE: 23
SKIP_NO_RECORDING: 20
validation_ok: true
```

There were `0` new enqueue decisions in the final pass because the previous
live pass had already seen the recording-backed calls. This proves the scheduler
does not create duplicate shadow capture decisions across repeated polls.

The previous live pass produced:

```text
source_rows: 44
ENQUEUE_SHADOW_CAPTURE: 21
SKIP_DUPLICATE: 3
SKIP_NO_RECORDING: 20
normalization_errors: 0
```

## Important audit fix

During self-audit, the seen-key logic was tightened:

- previous `ENQUEUE_SHADOW_CAPTURE` decisions are treated as seen;
- previous `SKIP_NO_RECORDING` decisions are not treated as permanently seen.

Reason: Mango recordings can appear with delay. If a no-recording call were
marked permanently seen, a later poll could miss the moment when the recording
becomes available.

## Runtime audit

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/scheduler_runtime_stage4_audit.json
```

Result:

```text
validation_ok: true
job_rows: 5
terminal_job_rows: 5
failed_job_rows: 0
blocked_job_rows: 0
retry_wait_job_rows: 0
running_job_rows: 0
status_counts:
  shadow_poll|succeeded: 5
warnings: 3
```

## Product DB integrity

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_integrity_stage4_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
schema_migrations: 3
product_calls: 297
job_runs: 5
due_job_runs: 0
running_job_runs: 0
failed_job_runs: 0
warnings: 3
```

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_live_shadow_poll.py \
  tests/test_productization_scheduler_runtime.py \
  tests/test_productization_mango_office_client.py \
  tests/test_productization_mango_shadow_poll_script.py
```

Result:

```text
27 passed, 1 warning
```

Stage 4 focused tests after the audit fix:

```text
17 passed, 1 warning
```

Full productization regression:

```text
120 passed, 1 warning
```

Insight-adjacent regression:

```text
32 passed
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Remaining limitations

This is still a shadow poll. It does not persist new calls into `product_calls`
and does not download recordings.

The next safe stage should add a product capture inbox table or JSONL-backed
inbox under the same product appliance root, so `ENQUEUE_SHADOW_CAPTURE` becomes
a durable queue item without touching the current processing runtime.
