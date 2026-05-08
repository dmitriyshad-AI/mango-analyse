# SaaS productization pass audit

Date: 2026-05-07

Scope: one safe pass across the five requested SaaS steps.

## Executive result

All five steps were completed as a safe productization slice:

1. Tenant-owned CRM owner mapping draft.
2. Read-only `ProductRepository` over the disposable SQLite DB.
3. Supervisor/scheduler dry-run skeleton.
4. UI/API data contracts and dashboard sample.
5. Insight seed layer with evidence refs.

This is not yet a production multi-tenant SaaS runtime. It is the first coherent
SaaS control surface around the Mango capture package, with tests and real
audits, while preserving the current processing pipeline.

## Safety boundary

This pass did not:

- change or delete `stable_runtime` DB/audio/transcripts
- run ASR
- run R+A
- write to AMO
- write to Tallanto
- change current batch/start/run-ui scripts

Writes were limited to:

- new isolated productization modules
- new isolated tests
- docs
- JSON reports under:
  `_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/`

## New modules

```text
src/mango_mvp/productization/repository.py
src/mango_mvp/productization/tenant_owner_mapping.py
src/mango_mvp/productization/supervisor.py
src/mango_mvp/productization/ui_contracts.py
src/mango_mvp/productization/insight_seed.py
```

New aggregate CLI:

```text
scripts/mango_office_saas_productization_audit.py
```

## Real command

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_saas_productization_audit.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --allowed-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest \
  --raw-payload _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl \
  --audio-dir _local_archive_mango_api_downloads_20260507/quarantine_import/audio \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/saas_productization_pass_audit.json \
  --call-limit 50 \
  --max-evidence-per-manager 3
```

Result:

```text
validation_ok: true
```

## Step 1: tenant owner mapping

Output:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/tenant_owner_mapping_draft.json
```

Result:

```text
manager_extensions: 8
confirmed_candidates: 5
manual_decisions_required: 3
calls_confirmed: 219
calls_pending: 78
validation_ok: true
```

Manual decisions required:

| Extension | Mango user | Calls | Required action |
|---:|---|---:|---|
| 23 | Тропов Олег | 76 | set CRM owner |
| 335 | Холодилова Дарья | 1 | set CRM owner |
| 387 | Головченко Карина | 1 | set CRM owner |

Product implication: CRM owner routing becomes tenant config, not hardcoded
logic.

## Step 2: ProductRepository

Schema:

```text
product_repository_readonly_v1
```

Result:

```text
call_records: 297
provider_metadata_rows: 297
enriched_view_rows: 297
manager_extensions: 8
calls_with_manager_identity: 297
calls_with_crm_owner: 219
raw_payload_refs_present: 297
validation_ok: true
warnings: 3
```

Product implication: UI, supervisor, and insight code can use one read-only
repository instead of opening runtime pipeline internals.

## Step 3: supervisor dry-run

Output:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/supervisor_dry_run_audit.json
```

Result:

```text
steps: 5
ready_steps: 4
warning_steps: 1
blocked_steps: 0
runtime_writes_allowed: false
asr_allowed: false
crm_writes_allowed: false
validation_ok: true
```

Step statuses:

| Step | Status | Evidence |
|---|---|---|
| poll_capture | ready | 665 raw payload rows |
| normalize_dedupe | ready | 297 provider metadata rows |
| archive_provenance | ready | 297 raw payload refs |
| quarantine_package | ready | 297 audio files |
| manager_identity | warning | 3 owner mappings need tenant review |

Product implication: the future appliance can show an operator-safe job plan
before any destructive or expensive work is enabled.

## Step 4: UI/API contracts

Output:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/ui_dashboard_contract_sample.json
```

Result:

```text
schema_version: saas_ui_contracts_v1
calls_sampled: 50
manager_filters: 8
manual_review_items: 3
```

Detailed contract doc:

```text
docs/SAAS_UI_DATA_CONTRACTS_2026-05-07.md
```

Product implication: first UI screen can be a real product dashboard: KPIs,
manager filters, call list, manual owner review queue, provenance drawer.

## Step 5: insight seed layer

Output:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/insight_seed_report.json
```

Result:

```text
schema_version: product_insight_seed_v1
seeds: 8
manager_volume_seeds: 5
manual_owner_seeds: 3
evidence_refs: 20
validation_ok: true
```

Seed examples:

| Seed | Topic | Manager | Calls | Priority |
|---|---|---|---:|---|
| insight-seed-0001 | manager_call_volume | Тютюнник Александр | 88 | medium |
| insight-seed-0002 | manual_crm_owner_mapping | Тропов Олег | 76 | high |
| insight-seed-0003 | manager_call_volume | Коршунова Анастасия | 58 | medium |

Product implication: insight UI can start with evidence-linked operational
cards now, and later attach ASR/R+A-derived conversation insights when that work
is explicitly enabled.

## Aggregate outputs

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/saas_productization_pass_audit.json
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/tenant_owner_mapping_draft.json
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/supervisor_dry_run_audit.json
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/ui_dashboard_contract_sample.json
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/insight_seed_report.json
```

## Tests

New focused test:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_saas_pass.py
```

Result:

```text
4 passed, 1 warning
```

The warning is the existing local `urllib3`/LibreSSL warning.

## Product assessment

This pass moves the project from "local processing scripts plus quarantine
artifacts" toward "appliance SaaS control plane":

- read-only product repository
- tenant configuration boundary
- job/supervisor audit boundary
- UI-ready contract
- evidence-linked insight seeds

The remaining blocker before CRM automation is not technical. It is the tenant
owner decision for 3 Mango managers, primarily extension `23` because it covers
76 calls.
