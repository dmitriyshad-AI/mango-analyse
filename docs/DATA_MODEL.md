# Data Model

Дата: 2026-05-09

Назначение: зафиксировать текущие модели данных Mango Analyse перед развитием
новых SaaS/productization фич. Документ описывает уже существующие контракты и
границы владения, а не проектирует новую схему с нуля.

## Data domains

| Domain | Storage | Owner | Write policy |
|---|---|---|---|
| Historical processing runtime | SQLite + files under `stable_runtime` and processing folders | processing dialog | Do not mutate from SaaS/productization dialog |
| Product appliance DB | isolated SQLite product DB | SaaS/productization dialog | Allowed only through product modules and path guards |
| Mango raw/capture artifacts | JSON/JSONL/report files under product roots | SaaS/productization dialog | Report/staging writes only |
| amoCRM runtime | SQLAlchemy DB plus amoCRM API | CRM/runtime layer | Live writes require explicit confirmation |
| Tallanto context | external API/read exports | CRM/runtime layer | Read-only in current scope |
| Insight artifacts | CSV/JSON/Markdown reports | insights layer | Report writes only |
| Agent runtime preview | SQLAlchemy tables, opt-in | experimental runtime | Disabled by default |

## Historical runtime model

Primary SQLAlchemy model:

- `src/mango_mvp/models.py`
- table: `call_records`

Purpose:

- current call processing state;
- ASR transcripts;
- Resolve outputs;
- Analyze outputs;
- legacy sync status.

Key identity fields:

| Field | Meaning |
|---|---|
| `id` | local runtime integer id |
| `source_file` | unique local audio path |
| `source_filename` | audio filename |
| `source_call_id` | provider/export call id when known |

Call metadata:

| Field | Meaning |
|---|---|
| `phone` | client phone |
| `manager_name` | manager label from source/export |
| `direction` | inbound/outbound/internal/unknown |
| `started_at` | call start timestamp |
| `duration_sec` | call duration |

Processing statuses:

| Field | Meaning |
|---|---|
| `transcription_status` | ASR state |
| `resolve_status` | Resolve state |
| `analysis_status` | Analyze state |
| `sync_status` | legacy sync state |
| `dead_letter_stage` | failed terminal stage |
| `next_retry_at` | retry scheduling |

Payload fields:

| Field | Meaning |
|---|---|
| `transcript_text` | final transcript text |
| `transcript_manager` | manager-side transcript |
| `transcript_client` | client-side transcript |
| `transcript_variants_json` | primary/secondary ASR variants and merge metadata |
| `resolve_json` | Resolve output |
| `analysis_json` | structured analysis output |

CRM references:

| Field | Meaning |
|---|---|
| `amocrm_contact_id` | matched amoCRM contact id |
| `amocrm_lead_id` | matched amoCRM lead id |

Boundary:

- this table remains owned by the current processing pipeline;
- Product API must not update it;
- future UI can show derived state only through product/read-only contracts.

## Productization contracts

Primary file:

- `src/mango_mvp/productization/contracts.py`

### `TenantRef`

Tenant identity for product mode.

Fields:

- `tenant_id`
- `display_name`

Current internal tenant examples can be company/project labels. Future client
appliance installs should use separate tenant ids and config roots.

### `TelephonyCallEvent`

Normalized provider call event.

Fields:

- `tenant`
- `provider`
- `provider_call_id`
- `started_at`
- `ended_at`
- `direction`
- `client_phone`
- `manager_ref`
- `recording_ref`
- `recording_url`
- `raw_payload`

Stable key:

```text
event_key = tenant_id:provider:provider_call_id
```

This is the primary idempotency key for Mango capture.

### `RecordingAsset`

Reference to a recording asset.

Fields:

- `event_key`
- `uri`
- `content_type`
- `checksum_sha256`
- `size_bytes`

### `CaptureIngestCandidate`

Candidate for controlled capture/processing handoff.

Fields:

- `event_key`
- `tenant_id`
- `provider`
- `provider_call_id`
- `started_at`
- `direction`
- `audio_ref`
- `client_phone`
- `manager_ref`
- `raw_payload`

Important: candidate creation does not run ASR/R+A.

### CRM snapshots

`CrmContactSnapshot` and `CrmOutcomeSnapshot` are provider-neutral read models
for future CRM adapters.

## Product appliance DB

Primary file:

- `src/mango_mvp/productization/product_db.py`

Current schema version:

```text
product_appliance_sqlite_v1
```

Required migrations:

| Migration | Meaning |
|---|---|
| `20260507_001_product_appliance_base` | base product appliance schema |
| `20260507_002_config_history_retention` | config history and retention policies |
| `20260507_003_scheduler_runtime` | scheduler runtime columns/indexes |
| `20260507_004_capture_inbox` | capture inbox table |

### `tenants`

Purpose: product tenant registry.

Key fields:

- `tenant_id`
- `display_name`
- `status`
- `created_at`
- `updated_at`

### `provider_accounts`

Purpose: telephony provider account metadata.

Key fields:

- `tenant_id`
- `provider`
- `mode`
- `config_ref`

Secrets should not be stored directly here. Use config references.

### `crm_accounts`

Purpose: CRM provider account metadata.

Key fields:

- `tenant_id`
- `provider`
- `mode`
- `config_ref`

### `tenant_manager_owner_map`

Purpose: map telephony manager references to CRM owners.

Primary key:

```text
tenant_id + telephony_provider + manager_extension
```

Key fields:

- `mango_name`
- `mango_email`
- `crm_provider`
- `crm_owner_id`
- `crm_owner_name`
- `crm_owner_email`
- `decision_status`
- `match_status`
- `source_ref`
- `config_ref`

This table drives manual owner review and prevents blind CRM ownership mapping.

### `product_calls`

Purpose: product-level call index derived from safe imports, not live runtime
mutation.

Primary key:

```text
tenant_id + telephony_provider + provider_call_id
```

Unique id:

```text
event_key
```

Key fields:

- `recording_id`
- `source_filename`
- `started_at`
- `duration_sec`
- `manager_extension`
- `manager_display_name`
- `crm_owner_id`
- `crm_owner_name`
- `crm_match_status`
- `raw_payload_ref`
- `source_repository_ref`

### `capture_inbox_items`

Purpose: product capture queue after shadow poll decisions.

Unique key:

```text
tenant_id + provider + event_key
```

Key fields:

- `status`
- `source_job_run_id`
- `source_report_ref`
- `raw_payload_ref`
- `started_at`
- `ended_at`
- `direction`
- `client_phone`
- `manager_ref`
- `recording_ref`
- `recording_url`
- `audio_ref`
- `decision_reason`
- `candidate_json`
- `event_json`
- `first_seen_at`
- `last_seen_at`
- `enqueue_count`
- `reserved_by`
- `reserved_at`
- `error`

Current important status:

- `ready_for_capture`

### `job_types` and `job_runs`

Purpose: scheduler/supervisor state.

Important statuses:

- `planned`
- `running`
- `succeeded`
- `retry_wait`
- `failed`
- `blocked`
- `skipped`

These tables are the future base for appliance supervision UI.

### `tenant_config_history`

Purpose: immutable snapshots of tenant config decisions.

Used for:

- owner mapping history;
- auditability;
- safe rollback of config changes.

### `retention_policies`

Purpose: local retention policy registry.

Default policy examples:

- product DB backups: review delete after 30 days;
- audit JSON: review archive after 180 days;
- tenant config history: keep for at least 3 years;
- product calls: manual review only.

## Product API data contracts

Primary file:

- `src/mango_mvp/productization/product_api.py`

Contract version:

```text
product_api_readonly_v1
```

Facade methods:

| Method | Product route concept | Data source |
|---|---|---|
| `dashboard_summary()` | `GET /dashboard/summary` | product DB snapshot |
| `capture_recent()` | `GET /capture/recent` | `capture_inbox_items` |
| `scheduler_runs()` | `GET /scheduler/runs` | `job_runs` |
| `asr_gate_status()` | `GET /asr/gates` | product ASR approval artifacts |
| `writeback_previews()` | `GET /writeback/previews` | product DB and policy |
| `processing_queue()` | `GET /queues/processing` | `capture_inbox_items` |
| `knowledge_playbook()` | `GET /knowledge/playbook` | schema-only placeholder |
| `settings_adapters()` | `GET /settings/adapters` | product config policy |
| `saas_stage_gates()` | `GET /saas/stage-gates` | stage gate report |

Actions policy:

- read-only methods are allowed;
- `download_audio`, `run_asr`, `run_ra`, `write_crm`, `write_runtime_db` are
  blocked in UI v1.

## UI data contract

Primary file:

- `src/mango_mvp/productization/ui_contracts.py`

Contract version:

```text
saas_ui_contracts_v1
```

Top-level shape:

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

Important DTOs:

- `CallListItemDTO`
- `ManagerFilterDTO`
- `ManualReviewDTO`

Stable UI key:

```text
event_key
```

Provenance field:

```text
raw_payload_ref
```

## amoCRM runtime model

Primary files:

- `src/mango_mvp/amocrm_runtime/models.py`
- `src/mango_mvp/amocrm_runtime/agent_models.py`

### `amo_integration_connections`

Purpose:

- OAuth/external amoCRM connection state;
- token/cache/catalog metadata;
- readiness and reauthorization status.

Important fields:

- `integration_mode`
- `status`
- `account_base_url`
- `account_subdomain`
- `client_id`
- `client_secret`
- `access_token`
- `refresh_token`
- `expires_at`
- `authorized_at`
- `last_error`
- `contact_field_catalog`
- `contact_field_catalog_synced_at`

### Agent preview tables

Opt-in only:

- `agent_runs`
- `agent_action_policies`
- `agent_actions`

Purpose:

- preview future agent actions;
- store policy, autonomy level, blockers and dry-run result.

Disabled by default unless:

```text
AI_OFFICE_AGENT_RUNTIME_ENABLED=1
```

## AMO/Tallanto field model

Canonical policy:

- `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`

Allowed contact write fields:

- `Статус матчинга`
- `AI-приоритет`
- `AI-рекомендованный следующий шаг`
- `Последняя AI-сводка`
- `Авто история общения`

Protected contact fields:

- `Id Tallanto`
- `Филиал Tallanto`

Allowed deal write fields:

- `AI-вердикт по закрытию`
- `AI-risk: premature close`
- `AI-основание вердикта`
- `AI-рекомендованный следующий шаг`
- `AI-дата следующего касания`
- `AI-сводка по сделке`

Live write requirement:

```text
execute_live_write=true and live_confirmation=WRITE_AMO_LIVE
```

## Insight data model

Primary package:

- `src/mango_mvp/insights/`

Current conceptual entities:

- client chain;
- call row;
- customer signal;
- manager answer pattern;
- outcome link;
- response quality score;
- playbook item;
- ROP validation item.

Current outputs are report artifacts, not transactional product DB tables.
This is acceptable until the first dashboard/knowledge feature needs live
querying from product DB.

## Future migration notes

SQLite remains acceptable for the current client-hosted appliance phase because:

- deployment is simple;
- backup is simple;
- only one local product writer is expected;
- historical processing already works this way.

Move active product queues to PostgreSQL only when at least one is true:

- multiple concurrent writers;
- multiple users operating the same appliance;
- centralized hosting for several clients;
- need for stronger locks, monitoring and online backup;
- product DB grows beyond comfortable local operational use.

Do not migrate historical runtime data just to say "SaaS". Migrate active
operational tables first.
