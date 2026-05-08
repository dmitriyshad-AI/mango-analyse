# SaaS UI data contracts

Date: 2026-05-07

Scope: productization branch, read-only over disposable quarantine DB.

## Goal

The first SaaS UI must not depend on runtime pipeline internals. It should read
stable product-facing contracts:

- repository summary
- manager filters
- call list items
- manual owner review queue
- provenance refs
- allowed and blocked actions

Implemented source:

```text
src/mango_mvp/productization/ui_contracts.py
```

Real sample:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/ui_dashboard_contract_sample.json
```

## Contract version

```text
saas_ui_contracts_v1
```

## Top-level shape

```json
{
  "schema_version": "saas_ui_contracts_v1",
  "summary": {},
  "filters": {},
  "views": {},
  "actions": {},
  "provenance": {}
}
```

## Summary

The `summary` block comes from `ProductRepositorySummary`:

- `call_records`
- `provider_metadata_rows`
- `enriched_view_rows`
- `manager_extensions`
- `calls_with_manager_identity`
- `calls_with_crm_owner`
- `manual_owner_review_items`
- `raw_payload_refs_present`
- `validation_ok`
- `blocked`
- `warnings`

Real values from the current quarantine DB:

```text
call_records: 297
provider_metadata_rows: 297
enriched_view_rows: 297
manager_extensions: 8
calls_with_manager_identity: 297
calls_with_crm_owner: 219
manual_owner_review_items: 3
raw_payload_refs_present: 297
validation_ok: true
```

## Filters

`filters.managers[]` is built from `manager_identity_map`.

Fields:

- `manager_extension`
- `label`
- `call_count`
- `crm_owner_status`: `present` or `missing`
- `crm_owner_id`
- `crm_owner_name`

This allows the UI to filter by Mango manager without knowing Mango API payload
shape or AMO user payload shape.

## Call List

`views.call_list.items[]` is the primary dashboard row contract.

Fields:

- `event_key`
- `source_filename`
- `started_at`
- `duration_sec`
- `provider`
- `provider_call_id`
- `recording_id`
- `manager_extension`
- `manager_display_name`
- `manager_crm_owner_id`
- `manager_crm_owner_name`
- `manager_crm_match_status`
- `raw_payload_ref`

The stable primary key for UI drill-down is:

```text
event_key
```

The provenance field is:

```text
raw_payload_ref
```

## Manual Owner Review Queue

`views.manual_owner_review_queue.items[]` contains tenant actions required before
CRM write automation.

Current queue:

| Extension | Mango user | Calls | Required action |
|---:|---|---:|---|
| 23 | Тропов Олег | 76 | set_or_confirm_crm_owner |
| 335 | Холодилова Дарья | 1 | set_or_confirm_crm_owner |
| 387 | Головченко Карина | 1 | set_or_confirm_crm_owner |

## Actions

Allowed in this pass:

- `shadow_poll_dry_run`
- `tenant_owner_mapping_review`
- `export_json_report`

Blocked in this pass:

- `download_audio`
- `run_asr`
- `run_ra`
- `write_crm`
- `write_runtime_db`

## UI implication

The first UI screen can now be a product dashboard, not a pipeline admin page:

1. Top KPI band from `summary`.
2. Manager filter from `filters.managers`.
3. Call table from `views.call_list`.
4. Manual review queue from `views.manual_owner_review_queue`.
5. Provenance drawer using `event_key` and `raw_payload_ref`.

No UI component needs direct access to `stable_runtime`, ASR state, or AMO write
code.
