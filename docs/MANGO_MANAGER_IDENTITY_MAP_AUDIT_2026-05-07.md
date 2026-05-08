# Mango manager identity map audit

Date: 2026-05-07

Scope: SaaS/productization branch only.

Goal: connect Mango `manager_extension` values from the quarantine capture sidecar
to human Mango users and, where possible, to AMO CRM owners. This makes captured
calls usable for ownership, UI filters, CRM-routing decisions, and future insight
dashboards without touching the current runtime pipeline.

## Safety boundary

This step did not:

- change `stable_runtime` DB/audio/transcripts
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- change current batch/start/run-ui scripts

Writes were limited to:

- disposable quarantine SQLite DB:
  `_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite`
- JSON/CSV audit artifacts under:
  `_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/`
- isolated productization code and tests

## Inputs

Disposable provider sidecar:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite
```

Mango users config snapshot:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/mango_users_config_20260507.json
```

AMO users snapshot:

```text
prod_runtime_transfer/data_handoff/live_export/users.json
```

## Implementation

Added an isolated module:

```text
src/mango_mvp/productization/manager_identity.py
```

It creates:

- table: `manager_identity_map`
- view: `provider_call_metadata_with_manager`

The table is keyed by:

```text
tenant_id, provider, manager_extension
```

The view enriches every row from `provider_call_metadata` with:

- `manager_display_name`
- `manager_email`
- `manager_crm_owner_id`
- `manager_crm_owner_name`
- `manager_crm_match_status`
- `manager_mapping_status`

Matching rules:

1. Mango user is matched by `telephony.extension`.
2. AMO owner is matched by exact normalized email.
3. If email does not match, AMO owner is matched by exact normalized name.
4. Missing AMO owner is a warning, not a block.
5. Missing Mango user is a block, because ownership cannot be trusted.

## Real run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_manager_identity_map.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --mango-users _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/mango_users_config_20260507.json \
  --amo-users prod_runtime_transfer/data_handoff/live_export/users.json \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_audit.json \
  --csv-out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_map.csv \
  --replace
```

Idempotency command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_manager_identity_map.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --mango-users _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/mango_users_config_20260507.json \
  --amo-users prod_runtime_transfer/data_handoff/live_export/users.json \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_idempotency_audit.json \
  --csv-out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_map.csv
```

## Result

Summary:

```text
validation_ok: true
blocked: 0
warnings: 3
sidecar_rows: 297
view_rows: 297
manager_extensions: 8
mapped_mango_users: 8
missing_mango_users: 0
crm_owner_matched: 5
crm_owner_unmatched: 3
calls_with_mango_user: 297
calls_with_crm_owner: 219
crm_owner_unmatched_call_count: 78
```

Counts by manager:

| Extension | Mango user | AMO owner | Calls | Match status |
|---:|---|---|---:|---|
| 26 | Тютюнник Александр | 13442054 / Тютюнник Александр | 88 | matched_name |
| 23 | Тропов Олег | missing | 76 | unmatched |
| 19 | Коршунова Анастасия | 13442050 / Коршунова Анастасия | 58 | matched_name |
| 202 | Тропина Анна | 13442066 / Леонова Анна | 35 | matched_email |
| 156 | Леонов Алексей | 13442070 / Леонов Алексей | 22 | matched_email |
| 105 | Козлова Екатерина | 13442058 / Козлова Екатерина | 16 | matched_email |
| 335 | Холодилова Дарья | missing | 1 | unmatched |
| 387 | Головченко Карина | missing | 1 | unmatched |

Manual review items:

| Extension | Mango user | Email | Calls | Reason |
|---:|---|---|---:|---|
| 23 | Тропов Олег | otrpvsaran@mail.ru | 76 | CRM owner unmatched |
| 335 | Холодилова Дарья | daria.vldmrvna@gmail.com | 1 | CRM owner unmatched |
| 387 | Головченко Карина | karinaglvchenko@gmail.com | 1 | CRM owner unmatched |

Interpretation:

- Mango ownership is complete for the quarantine capture set: 297/297 calls.
- CRM ownership is usable for 219/297 calls immediately.
- 78 calls need a business mapping decision, not a pipeline fix.
- The most important manual decision is extension `23` / Тропов Олег, because it covers 76 calls.

## Artifacts

Primary audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_audit.json
```

Idempotency audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_idempotency_audit.json
```

CSV mapping table:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/manager_identity_map.csv
```

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_manager_identity.py \
  tests/test_productization_manager_identity_script.py
```

Result:

```text
6 passed, 1 warning
```

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
82 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning. It is not related
to the manager identity layer.

## Next decision

Before using AMO ownership for automated CRM writes, resolve the unmatched CRM
owner mappings:

- extension `23` / Тропов Олег
- extension `335` / Холодилова Дарья
- extension `387` / Головченко Карина

For productization, these should become tenant-owned configuration rows, not
hardcoded rules.
