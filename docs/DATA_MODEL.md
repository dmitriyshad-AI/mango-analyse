# Data Model

Дата: 2026-07-19

Назначение: зафиксировать модели данных, которые реально используются сейчас.

## Data domains

| Domain | Storage | Owner | Write policy |
|---|---|---|---|
| Historical processing runtime | SQLite + files under `stable_runtime` and processing folders | calls pipeline | Single-writer; do not mutate from bot tasks |
| Mango raw/capture artifacts | JSON/JSONL/report files under the calls pipeline root | calls pipeline | Report/staging writes only |
| amoCRM runtime | SQLAlchemy DB plus amoCRM API | CRM/runtime layer | Live writes require explicit confirmation |
| Tallanto context | external API/read exports | CRM/runtime layer | Read-only in current scope |
| Agent runtime preview | SQLAlchemy tables, opt-in | experimental runtime | Disabled by default |

Question Catalog treats
`stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v11_frozen_gate/enriched_reviews.csv`
as a retained read-only input: 2,726 rows, SHA-256
`d71ac22699f04c67ba3f464c7cfb4886cc3c73d046717a16f5a045ec2ecf7270`.
It is not regenerated during normal builds; a missing file blocks the catalog
instead of silently dropping the call channel.

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
- bot and reporting tasks must not update it.

## Runtime adapter models

The historical SaaS/appliance layer was retired in July 2026. The remaining
mango_mvp.productization package is a legacy namespace for three live
integration boundaries:

- contracts.py: normalized Mango call events and recording references;
- mail_archive.py / mail_imap_snapshot.py: canonical read-only mail archive;
- product_db.py / tallanto_snapshot_exporter.py: guarded local index used by
  the optional read-only Tallanto snapshot.

TelephonyCallEvent.event_key remains the idempotency key:

    tenant_id:provider:provider_call_id

These adapters do not define a reusable SaaS product, tenant UI, scheduler, or
product API. Those retired contracts were removed from the current tree.

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

## Future migration notes

SQLite remains acceptable for the current local runtime because:

- deployment is simple;
- backup is simple;
- only one local writer is expected;
- historical processing already works this way.

Move active queues to PostgreSQL only when at least one is true:

- multiple concurrent writers;
- multiple users operate the same runtime;
- centralized hosting is required;
- need for stronger locks, monitoring and online backup;
- the active DB grows beyond comfortable local operation.

There is no active SaaS product DB or migration project. Do not migrate
historical data without a concrete business need.
