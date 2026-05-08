# SaaS ASR Worker Sandbox Execution Request Stage 25 Audit

Date: 2026-05-08

## Scope

Stage 25 adds a request-only dry run for ASR sandbox execution. It validates:

- Stage 23 ASR sandbox approval packet.
- Stage 21 ASR sandbox execution contract.
- Optional Stage 24 human approval record.

The stage writes only Stage 25 request/audit JSON files. It does not dispatch a worker, run ASR/R+A, create sandbox output/tmp directories, write transcripts, write runtime DBs, or write CRM/Tallanto.

## Implementation

Added:

- `src/mango_mvp/productization/asr_worker_sandbox_execution_request.py`
- `scripts/mango_office_asr_worker_sandbox_execution_request.py`
- `tests/test_productization_asr_worker_sandbox_execution_request.py`

Updated:

- `src/mango_mvp/productization/__init__.py`

## Real Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_asr_worker_sandbox_execution_request.py
```

Result:

- `validation_ok=true`
- `execution_request_ready=false`
- `approval_packet_valid=true`
- `contract_valid=true`
- `approval_record_present=false`
- `approval_record_valid=false`
- `tasks=21`
- `requested_tasks=0`
- `selected_engine=mlx`
- action: `BLOCK_ASR_SANDBOX_EXECUTION_REQUEST_PENDING_HUMAN_APPROVAL`
- reason: `approval_record_missing`
- `dispatch_allowed=false`
- `run_asr=false`
- `write_transcripts=false`

Artifact paths:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_execution_request_stage25/asr_worker_sandbox_execution_request_stage25.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_execution_request_stage25/asr_worker_sandbox_execution_request_stage25_audit.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_execution_request_stage25/asr_worker_sandbox_execution_request_stage25_idempotency_audit.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_execution_request_stage25/product_db_integrity_stage25_audit.json`

SHA256:

- request: `76e5cfb1f30646899d22f890352c36a3b1b9e3fee36c91372502ce17fee774b4`
- audit: `a7670f68c8013767c5095e1c12962953dad6ec05a5d21706e29357509828abc5`
- idempotency audit: `c3dfcf34ddf94551caf64cd345e61e002f5a9f4fe091d57e6255f3b01cf0869c`
- product DB integrity audit: `2916d189e338478cd8c41239e70269ab634cccffe7b447776cb1f0f7caed2fa2`

## Safety

Confirmed in real run:

- `creates_sandbox_output_dirs=false`
- `creates_sandbox_tmp_dirs=false`
- `dispatch_worker=false`
- `reads_audio=false`
- `run_asr=false`
- `run_ra=false`
- `write_transcripts=false`
- `runtime_db_writes=false`
- `stable_runtime_writes=false`
- `write_crm=false`
- `write_tallanto=false`

## Verification

Commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile src/mango_mvp/productization/asr_worker_sandbox_execution_request.py scripts/mango_office_asr_worker_sandbox_execution_request.py src/mango_mvp/productization/__init__.py
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_asr_worker_sandbox_execution_request.py
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_asr_worker_sandbox_execution_request.py tests/test_productization_asr_worker_sandbox_human_approval_record.py tests/test_productization_asr_worker_sandbox_approval_packet.py tests/test_productization_asr_worker_sandbox_preflight.py tests/test_productization_asr_worker_sandbox_execution_contract.py tests/test_productization_asr_worker_sandbox_readiness.py tests/test_productization_asr_worker_execution_dry_run.py tests/test_productization_asr_execution_plan.py
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py integrity --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_execution_request_stage25/product_db_integrity_stage25_audit.json
```

Results:

- New focused tests: `5 passed`
- Stage 18-25 focused chain: `46 passed`
- Full productization tests: `229 passed, 1 warning`
- Product DB integrity: `validation_ok=true`, `product_calls=297`, `capture_inbox_ready=21`, `warnings=3`

The warning is external: `urllib3` reports that the local Python SSL module is compiled with LibreSSL.

## Audit Notes

The real Stage 25 run is intentionally blocked because the Stage 24 human approval record does not exist. This is the correct state for the current thread constraints: no ASR execution approval has been given, and no ASR worker was dispatched.

When a valid Stage 24 human approval record exists, Stage 25 can build `PLAN_ASR_SANDBOX_EXECUTION_REQUEST_NOT_DISPATCHED` with request items, but still keeps `dispatch_allowed=false`, `run_asr=false`, `create_dirs=false`, and `write_transcripts=false`.

## Next Step

Stage 26 should create a final launcher dry-run/audit layer that consumes a ready Stage 25 request and proves exactly what command would be dispatched, but still does not run ASR unless the user gives a separate explicit execution approval.
