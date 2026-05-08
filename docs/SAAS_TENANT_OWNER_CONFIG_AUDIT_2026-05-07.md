# SaaS tenant owner config audit

Date: 2026-05-07

Scope: Stage 1 of the SaaS/productization plan: close tenant owner config.

## Result

The technical owner-config workflow is implemented and verified:

- dry-run validation
- strict blocking for incomplete owner rows
- apply mode for the isolated product DB
- synchronized updates to `tenant_manager_owner_map` and `product_calls`
- audit JSON outputs
- tests for blocked config and successful complete config apply

The current real tenant config is intentionally not applied because 3 CRM owner
assignments are still missing. No fallback owner was invented.

## Safety boundary

This step did not:

- change `stable_runtime`
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- write to the runtime processing DB

Apply mode writes only to:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite
```

The real incomplete config was tested in apply mode and correctly wrote nothing.

## New implementation

Updated module:

```text
src/mango_mvp/productization/product_db.py
```

New CLI:

```text
scripts/mango_office_product_owner_config.py
```

Updated tests:

```text
tests/test_productization_product_db.py
```

## Real config

Path:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json
```

Schema:

```text
tenant_owner_mapping_v1
```

Current status:

```text
config_entries: 8
complete_owner_entries: 5
missing_owner_entries: 3
would_confirm_existing: 5
would_set_owner: 0
blocked: 3
validation_ok: false
```

## Real dry-run command

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_owner_config.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_dry_run_audit.json
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_dry_run_audit.json
```

Result:

```text
validation_ok: false
blocked: 3
pending_owner_mappings_before: 3
pending_owner_mappings_after: 3
calls_with_crm_owner_before: 219
calls_with_crm_owner_after: 219
```

## Real apply-mode blocked check

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_owner_config.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_apply_blocked_audit.json \
  --apply
```

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_apply_blocked_audit.json
```

Result:

```text
validation_ok: false
applied: 0
blocked: 3
product_db_writes: false
```

DB remained unchanged:

```text
tenant_manager_owner_map:
  confirmed_candidate: 5
  needs_manual_owner: 3

product_calls:
  has_owner: 219
  missing_owner: 78
```

## Required business decisions

The following config entries must receive `crm_owner_id` and `crm_owner_name`.
`crm_owner_email` is optional but recommended.

| Extension | Mango user | Mango email | Calls | Required fields |
|---:|---|---|---:|---|
| 23 | Тропов Олег | otrpvsaran@mail.ru | 76 | `crm_owner_id`, `crm_owner_name` |
| 335 | Холодилова Дарья | daria.vldmrvna@gmail.com | 1 | `crm_owner_id`, `crm_owner_name` |
| 387 | Головченко Карина | karinaglvchenko@gmail.com | 1 | `crm_owner_id`, `crm_owner_name` |

Current AMO users snapshot contains these possible owner IDs:

| AMO owner ID | Name | Email |
|---:|---|---|
| 11837118 | Админ | integration@kmipt.ru |
| 13442034 | Клычева Дарья | dasha.klycheva00@mail.ru |
| 13442050 | Коршунова Анастасия | nastasya_knopa@mail.ru |
| 13442054 | Тютюнник Александр | kryrrogg2015@gmail.com |
| 13442058 | Козлова Екатерина | katerina1851@mail.ru |
| 13442066 | Леонова Анна | tropina-ann@mail.ru |
| 13442070 | Леонов Алексей | limon-999@inbox.ru |

No automatic match exists for extensions `23`, `335`, and `387` in this AMO
snapshot. Assigning them to `Админ` or another employee is a business decision,
not an engineering inference.

## How to finish Stage 1 after owner decision

Edit:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json
```

For each of the 3 rows, fill:

```json
{
  "crm_owner_id": 13442000,
  "crm_owner_name": "Имя владельца",
  "crm_owner_email": "email@example.com",
  "confirmed_by": "who_approved",
  "notes": "business owner mapping decision"
}
```

Then run dry-run:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_owner_config.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_dry_run_audit.json
```

Expected after filling all owners:

```text
validation_ok: true
blocked: 0
would_set_owner: 3
calls_would_gain_owner: 78
pending_owner_mappings_after: 3
```

Then apply:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_product_owner_config.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --config _local_archive_mango_api_downloads_20260507/product_appliance/config/tenant_owner_mapping_foton_mango.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/tenant_owner_config_apply_audit.json \
  --apply
```

Expected after apply:

```text
validation_ok: true
pending_owner_mappings_after: 0
calls_with_crm_owner_after: 297
```

## Verification

Targeted tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_db.py
```

Result:

```text
10 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Stage 1 conclusion

Stage 1 is technically complete and business-blocked:

- the software now enforces the tenant owner config correctly
- incomplete config cannot be applied
- apply mode has been verified not to write on invalid config
- the remaining work is a human owner assignment for 3 Mango extensions
