# SaaS/productization architecture plan

Дата: 2026-05-07

Цель: зафиксировать безопасную SaaS/productization ветку, которая не конфликтует с текущей обработкой звонков, не трогает runtime-БД/аудио/транскрипты и готовит Mango Analyse к продуктовой эксплуатации.

## Входной контекст

Прочитанные документы:

- `docs/SAAS_AND_INSIGHT_ROADMAP_2026-05-07.md`
- `docs/PROJECT_CLEANUP_AUDIT_2026-05-07.md`
- `docs/OPERATIONS_RUNBOOK_2026-05-07.md`
- `docs/RUNTIME_RETENTION_POLICY_2026-05-07.md`
- `TZ_FULL_VISION_2026-05-01.md`

Текущий operational gate:

- Актуальный coverage: `stable_runtime/final_processing_coverage_report_20260507_v4/`
- Actionable ASR gap: `0`
- Остаток Resolve+Analyze: `468` manual tails
- Manager-manager no-ASR звонки сейчас исключены и не распознаются
- Следующий основной поток: закрыть R+A tails, затем contact-layer/ROP/AMO-ready

Ограничение этой ветки: только docs, новая архитектура, изолированные модули и тесты. Никаких ASR/R+A запусков, AMO/Tallanto writes, изменений batch/start/run-ui scripts и записей в `stable_runtime`.

## Продуктовая траектория

### Этап 1. Автономный сервис для своей компании

Система должна сама подтягивать новые звонки, ставить их в обработку, готовить саммари/приоритеты, давать РОПу очередь действий и писать в CRM только через dry-run/staged gates. На этом этапе допустимо, что часть старого batch pipeline остается SQLite/local, но новый operational контур уже проектируется как server-ready.

Критерий ценности: после звонка менеджер и РОП получают структурированную информацию без ручного экспорта Mango.

### Этап 2. Демонстрационный продукт

Нужны стабильные data contracts, dashboard, ROP queue, AMO writeback preview, Knowledge Lab и демо-tenant с обезличенными данными. Клиентам показывается не набор скриптов, а repeatable workflow: подключили телефонию/CRM, увидели звонки, получили аналитику и очереди действий.

Критерий ценности: систему можно показать как управляемый продукт без доступа к приватному runtime.

### Этап 3. Ограниченный multi-tenant SaaS

Добавляются tenant isolation, secrets management, storage retention, audit log, SLA/alerts, deploy runbook и adapter layer для других телефоний/CRM. Core analysis не должен знать, откуда пришел звонок: Mango/AMO являются первыми адаптерами, но не архитектурным центром.

Критерий ценности: второй клиент подключается конфигурацией и адаптерами, а не форком кода.

## Целевая архитектура

```text
Telephony adapters        Capture service         Operational store
Mango first          ->    normalized calls   ->   PostgreSQL active layer
Other PBX later           idempotency              raw payload audit
                           recording refs           queue/status tables

CRM adapters              Core pipeline bridge      Insight layer
AMO first            <-    existing ASR/R+A    ->   questions/answers
Tallanto context          writeback preview         outcomes/rubrics
Other CRM later           quality gate              validated KB

Scheduler/supervisor      Product UI
morning/hourly/evening    dashboard, coverage, batch builder,
health/retry/DLQ          ROP queue, writeback preview, knowledge lab
```

### Adapter layer

Telephony adapter contract:

- `poll_calls(tenant, since, until)` returns normalized call events.
- `get_recording(event)` resolves a recording reference or storage URI.
- It never triggers ASR directly.
- It stores raw provider payload for audit and later debugging.

CRM adapter contract:

- Reads contact/deal/outcome context by phone/time window.
- Builds writeback previews.
- Production writes are separate and require explicit staged gates.
- AMO and Tallanto are first adapters, not hardcoded core dependencies.

### Mango API polling/capture POC

Start with polling instead of webhook:

1. Verify exact Mango VPBX API methods and credentials in the account.
2. Poll a small recent window, for example last 2 hours.
3. Normalize provider payloads into `TelephonyCallEvent`.
4. Deduplicate by `tenant_id + provider + provider_call_id`.
5. Resolve recording reference/link, but run in shadow mode first.
6. Compare captured metadata with manual exports.
7. Only after shadow confidence, attach capture output to the existing pipeline bridge.

Reason: polling is easier to replay, easier to make idempotent and safer after downtime. Webhook can be added later as a latency optimization, with polling kept as recovery.

### SQLite -> PostgreSQL

Do not migrate the historical runtime now just to "be SaaS". Keep SQLite for:

- current local batch pipeline
- historical processing
- portable external worker packs

Introduce PostgreSQL for the new operational layer:

- tenants
- captured calls
- raw provider payloads
- queue/status rows
- CRM writeback audit
- scheduler runs
- feedback and notification state

Migration path:

1. Freeze data contracts and pipeline statuses.
2. Create Postgres-compatible repository interfaces.
3. Mirror new capture records into Postgres only.
4. Add a read bridge to existing SQLite results when needed.
5. Move active queues/writeback/audit first.
6. Migrate historical data later only when a product use case requires it.

### Supervisor/scheduler

Needed cycles:

- `poll_recent_calls`: every 5-15 minutes, tenant-aware.
- `catchup_calls`: on startup and after downtime, configurable lookback.
- `pipeline_bridge`: enqueue ready captured calls after storage is available.
- `crm_context_refresh`: hourly or daily for outcome/linking.
- `daily_rop_digest`: morning/evening summaries.
- `dead_letter_report`: daily failures, retries and stale queues.

Implementation rule: scheduler writes operational audit rows, not shell-only logs.

### UI redesign and data contracts

Redesign should wait for contracts, not for full backend completion. First UI contract endpoints:

- `GET /dashboard/summary`
- `GET /coverage/monthly`
- `GET /capture/recent`
- `GET /queues/rop`
- `GET /writeback/previews`
- `GET /knowledge/questions`
- `GET /scheduler/runs`
- `GET /settings/adapters`

UI screens:

- Dashboard
- Coverage
- Batch Builder
- Worker Control
- ROP Queue
- AMO Writeback
- Knowledge Lab
- Settings

Rule: UI does not call batch scripts directly once product mode begins. It calls product APIs that own policy, gates and audit.

### Knowledge/insight layer

First commercial insight layer:

1. Extract client questions.
2. Extract manager answers.
3. Link calls to outcomes from AMO/Tallanto.
4. Score answers by rubric: completeness, accuracy, objection handling, next step and compliance.
5. Control confounders: source/UTM, manager, course, client status, seasonality, price and number of touches.
6. Produce validated answer candidates for ROP review.
7. Feed approved knowledge into UI, ROP digest and future bot.

Risk rule: LLM scores are evidence candidates, not truth. Product reports must separate observed correlation from validated recommendation.

## Safe first implementation slice

Recommended first slice for this second dialog:

1. Add architecture doc and non-goals.
2. Add pure Python data contracts for tenant, telephony event, recording asset and CRM outcome.
3. Add adapter protocols for telephony and CRM.
4. Add capture planner that produces shadow ingest decisions from normalized events.
5. Add Mango payload mapper skeleton for polling POC.
6. Add tests with synthetic payloads.

Why this slice is safe:

- It does not import current ASR/R+A services.
- It does not open or mutate runtime DBs.
- It does not access audio files.
- It does not call AMO/Tallanto.
- It does not change launch scripts or current CLI.
- It gives future threads a stable contract for Mango, AMO, PostgreSQL and UI work.

Acceptance criteria:

- `PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py`
- Tests prove deterministic event keys, duplicate handling, missing-recording handling and Mango payload normalization.
- No files under `stable_runtime/` are changed.
- No existing pipeline files are modified.

## Workstreams after the first slice

### Stream A. Mango polling/capture POC

- `MangoOfficeClient` with verified auth/signature.
- Poll recent calls in read-only shadow mode.
- Persist raw payloads into a new disposable POC store, not current runtime DB.
- Compare with manual exports.
- Add recording download only after metadata matching is correct.

### Stream B. PostgreSQL operational schema

- Draft schema for tenants, captured calls, provider payloads, queue items and writeback audit.
- Add repository interfaces and SQLite-free tests.
- Add migration strategy document.
- Defer migration of historical call records.

### Stream C. Supervisor/scheduler

- Add scheduler contract and run records.
- Add dry-run in-process scheduler first.
- Later move to APScheduler/Celery/RQ only if the operational shape requires it.

### Stream D. UI contracts

- Define JSON contracts from backend to UI.
- Build mock API responses.
- Redesign UI against mocks before attaching live data.

### Stream E. Knowledge/insight layer

- Define question/answer/outcome/rubric schemas.
- Use completed transcript+analysis only.
- Start with offline exports and reports.
- Keep AMO/Tallanto reads dry and explicit.

## Decisions still required

- PostgreSQL hosting: managed Selectel/Yandex/other versus self-hosted.
- Storage: S3-compatible bucket immediately versus local MinIO first.
- Mango credentials and exact VPBX API endpoints in the customer account.
- Tenant model: one DB per tenant versus shared DB with tenant_id isolation.
- Secrets manager: `.env` for internal phase, sops/Doppler before external clients.
- Retention policy for SaaS tenants: audio, transcript, raw payload, derived insights.
- Legal/compliance package for personal data processing.

## Non-goals for this branch

- Closing `468` R+A tails.
- Recognizing manager-manager excluded calls.
- Rebuilding coverage/contact-layer/ROP/AMO-ready.
- Running ASR/R+A.
- Writing to AMO/Tallanto.
- Editing current batch/start/run-ui scripts.
- Cleaning or deleting runtime artifacts.
