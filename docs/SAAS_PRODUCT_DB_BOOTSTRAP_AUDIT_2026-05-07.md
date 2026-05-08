# SaaS product DB bootstrap audit

Date: 2026-05-07

Scope: tenant config plus first product appliance SQLite DB.

## Goal

Start the next recommended stage:

1. Make manager-to-CRM ownership tenant configuration, not hardcoded code.
2. Create a separate product appliance DB that can become the local SaaS runtime
   database while keeping the current processing/runtime DB untouched.

## Safety boundary

This step did not:

- change `stable_runtime`
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- change current batch/start/run-ui scripts

Writes were limited to:

- isolated productization code and tests
- docs
- product appliance artifacts under:
  `_local_archive_mango_api_downloads_20260507/product_appliance/`

## New implementation

Module:

```text
src/mango_mvp/productization/product_db.py
```

CLI:

```text
scripts/mango_office_product_db_bootstrap.py
```

Tests:

```text
tests/test_productization_product_db.py
```

Exports added through:

```text
src/mango_mvp/productization/__init__.py
```

## Product DB schema

Schema version:

```text
product_appliance_sqlite_v1
```

Base migration:

```text
20260507_001_product_appliance_base
```

Tables:

- `schema_migrations`
- `tenants`
- `provider_accounts`
- `crm_accounts`
- `tenant_manager_owner_map`
- `product_calls`
- `job_types`
- `job_runs`

Key design decisions:

- `product_calls` is keyed by `tenant_id, telephony_provider, provider_call_id`.
- `event_key` is unique.
- `tenant_manager_owner_map` is keyed by `tenant_id, telephony_provider, manager_extension`.
- Job types exist now, but active processing jobs remain disabled.
- The product DB stores a snapshot/projection from the quarantine repository, not
  the runtime processing DB.

## Tenant config

Generated config:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json
```

Schema:

```text
tenant_owner_mapping_v1
```

Current pending mappings:

| Extension | Mango user | Email | Calls | Decision |
|---:|---|---|---:|---|
| 23 | Тропов Олег | otrpvsaran@mail.ru | 76 | needs_manual_owner |
| 335 | Холодилова Дарья | daria.vldmrvna@gmail.com | 1 | needs_manual_owner |
| 387 | Головченко Карина | karinaglvchenko@gmail.com | 1 | needs_manual_owner |

The config also includes the 5 currently matched candidates so they can be
confirmed or overridden by the tenant later.

## Real command

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_bootstrap.py \
  --source-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --source-db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --tenant-config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_bootstrap_audit.json \
  --backup _local_archive_mango_api_downloads_20260507/product_appliance/backups/mango_product_appliance_20260507.sqlite \
  --replace
```

## Real outputs

Product DB:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite
```

Tenant config:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json
```

Audit:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_bootstrap_audit.json
```

Backup:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/backups/mango_product_appliance_20260507.sqlite
```

## Result

Integrity summary:

```text
validation_ok: true
blocked: 0
warnings: 3
schema_migrations: 1
tenants: 1
manager_owner_rows: 8
product_calls: 297
calls_with_crm_owner: 219
pending_owner_mappings: 3
raw_payload_refs_present: 297
job_types: 5
```

Owner status:

```text
confirmed_candidate: 5
needs_manual_owner: 3
```

Call owner status:

```text
has_owner: 219
missing_owner: 78
```

Job types:

| Job type | Default mode |
|---|---|
| shadow_poll | dry_run |
| capture_download | disabled |
| asr | disabled |
| ra | disabled |
| crm_sync | disabled |

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_db.py
```

Result:

```text
6 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning.

## What this unlocks

The project now has a local appliance DB boundary:

- tenant config can be reviewed and versioned
- UI can read `product_calls` and `tenant_manager_owner_map`
- scheduler can write `job_runs` later
- provider/CRM account rows have a home
- backup exists for the product DB artifact

The next blocking business action is still owner mapping for extension `23`.
After that, the product DB can represent `manual_owner_review_items = 0` for
the current Mango capture set.
