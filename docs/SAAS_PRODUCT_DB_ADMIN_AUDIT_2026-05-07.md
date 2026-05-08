# SaaS product DB admin audit

Date: 2026-05-07

Scope: Stage 2 of the SaaS/productization plan: strengthen the local product
appliance SQLite DB.

## Result

Stage 2 is complete.

The product appliance DB now has:

- idempotent upgrade/migration workflow
- separate integrity-check CLI
- backup CLI
- restore CLI
- review-only retention audit
- versioned tenant config history
- tests covering admin workflows

The work remains isolated to the product appliance branch and does not touch the
runtime processing pipeline.

## Safety boundary

This step did not:

- change `stable_runtime`
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- write to the runtime processing DB

Writes were limited to:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/
```

## New implementation

Updated module:

```text
src/mango_mvp/productization/product_db.py
```

New admin CLI:

```text
scripts/mango_office_product_db_admin.py
```

Updated tests:

```text
tests/test_productization_product_db.py
```

Updated exports:

```text
src/mango_mvp/productization/__init__.py
```

## New migration

Added migration:

```text
20260507_002_config_history_retention
```

Current migrations in the real product DB:

```text
20260507_001_product_appliance_base
20260507_002_config_history_retention
```

New tables:

- `tenant_config_history`
- `retention_policies`

## Real upgrade

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  upgrade \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_upgrade_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_upgrade_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
migrations_applied: 1
schema_migrations_before: 1
schema_migrations_after: 2
```

## Real integrity check

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_integrity_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_integrity_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
warnings: 3
schema_migrations: 2
tenants: 1
manager_owner_rows: 8
product_calls: 297
calls_with_crm_owner: 219
pending_owner_mappings: 3
raw_payload_refs_present: 297
job_types: 5
tenant_config_history: 1
retention_policies: 4
```

The 3 warnings are the known pending CRM owner mappings from Stage 1.

## Tenant config history

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  snapshot-config \
  --config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --reason stage2_admin_snapshot \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/tenant_config_snapshot_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/tenant_config_snapshot_audit.json
```

Result:

```text
validation_ok: true
inserted: 1
tenant_id: foton
config_kind: tenant_owner_mapping
content_hash: 7f50dd3e178e29c6d72ea842d7b5be0d1a850c71839d137a715e51fa9a5fe0b2
```

## Retention policies

Current policies:

| Target | Retention days | Action | Enabled |
|---|---:|---|---:|
| audit_json | 180 | review_archive | 1 |
| product_calls | 1095 | manual_review_only | 0 |
| product_db_backup | 30 | review_delete | 1 |
| tenant_config_history | 1095 | keep | 1 |

Retention audit command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  retention-audit \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_retention_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/product_db_retention_audit.json
```

Result:

```text
validation_ok: true
blocked: 0
policies: 4
enabled_policies: 3
artifacts_scanned: 12
review_candidates: 0
```

The retention audit is review-only:

```text
deletes_files: false
deletes_db_rows: false
review_only: true
```

## Backup and restore

Backup command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  backup \
  --backup _local_archive_mango_api_downloads_20260507/product_appliance/backups/mango_product_appliance_stage2.sqlite \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_backup_stage2_audit.json
```

Backup:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/backups/mango_product_appliance_stage2.sqlite
```

Result:

```text
validation_ok: true
size_bytes: 380928
```

Restore-check command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_db_admin.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/restore_check/mango_product_appliance.sqlite \
  restore \
  --backup _local_archive_mango_api_downloads_20260507/product_appliance/backups/mango_product_appliance_stage2.sqlite \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/product_db_restore_check_audit.json \
  --replace
```

Restore-check DB:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/restore_check/mango_product_appliance.sqlite
```

Result:

```text
validation_ok: true
blocked: 0
restored_size_bytes: 380928
product_calls: 297
tenant_config_history: 1
retention_policies: 4
```

The restore was tested against a separate `restore_check` DB, not by replacing
the main product DB.

## Admin CLI operations

The admin CLI now supports:

```text
integrity
upgrade
backup
restore
retention-audit
snapshot-config
```

All output paths are constrained under product root.

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_db.py
```

Result:

```text
16 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Stage 2 conclusion

Stage 2 is complete:

- the product DB can be upgraded idempotently
- integrity checks are standalone
- backups are created under product root
- restore is tested under product root
- retention rules exist and are review-only
- tenant config history is versioned by content hash

Remaining warnings are not Stage 2 defects. They are the known Stage 1 business
gap: 3 CRM owner mappings still need a human decision.
