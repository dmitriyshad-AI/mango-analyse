# Next Feature Plan: Appliance Dashboard v1

Дата: 2026-05-09

## Decision

Следующая новая фича: локальный read-only Appliance Dashboard поверх Product API.

Почему именно она:

- превращает уже сделанные productization слои в видимый продукт;
- не конфликтует с processing-диалогом;
- не требует ASR/R+A запусков;
- не пишет в AMO/Tallanto;
- дает понятную демо-поверхность для внутреннего использования и будущих
  клиентских показов.

## Feature goal

Открыть локальный web-интерфейс, где владелец продукта видит:

- состояние product DB;
- последние Mango capture events;
- capture inbox;
- scheduler/job runs;
- AMO writeback readiness;
- ASR gates;
- knowledge/insight readiness;
- adapter/settings safety state.

## Non-goals

- не запускать ASR/R+A из UI;
- не скачивать аудио одной кнопкой без existing controlled download flow;
- не писать в runtime DB;
- не писать в AMO/Tallanto;
- не строить полноценный multi-tenant cloud UI;
- не делать красивый SaaS marketing page.

## Current inputs

Architecture docs:

- `docs/ARCHITECTURE_CURRENT.md`
- `docs/DATA_MODEL.md`
- `docs/SAAS_UI_DATA_CONTRACTS_2026-05-07.md`
- `docs/SCRIPT_SAFETY_MATRIX.md`

Product API:

- `src/mango_mvp/productization/product_api.py`
- `src/mango_mvp/productization/product_api_http.py`

Data contracts:

- `product_api_readonly_v1`
- `saas_ui_contracts_v1`

## User experience v1

First screen should be operational, not a landing page.

Recommended layout:

1. Top status band:
   - product DB present;
   - validation status;
   - blocked count;
   - warnings;
   - capture ready count;
   - pending owner mappings.
2. Capture inbox:
   - recent events;
   - status;
   - manager;
   - recording ref;
   - raw payload ref;
   - enqueue count.
3. Scheduler:
   - latest job runs;
   - failed/blocked/retry_wait counts;
   - next planned jobs.
4. Writeback readiness:
   - current mode `preview_only`;
   - blocked reasons;
   - required rollout sequence.
5. Gates and safety:
   - ASR dispatch allowed: false;
   - run ASR: false;
   - write CRM: false;
   - write runtime DB: false.
6. Knowledge readiness:
   - current mode;
   - available insight artifacts;
   - blocked reasons for live playbook if any.
7. Settings:
   - telephony adapter;
   - CRM adapter;
   - DB profile;
   - credentials refs without secret values.

## Backend work packages

### WP-1. Product API HTTP hardening

Scope:

- verify existing HTTP layer exposes needed read-only routes;
- add missing read-only route wrappers if needed;
- ensure invalid query params return JSON errors;
- ensure unsupported mutation methods return 405/404 and never mutate.

Tests:

- `TestClient` or subprocess HTTP smoke tests;
- no real Mango/AMO/Tallanto network;
- temp product DB fixtures.

### WP-2. Demo product DB fixture

Scope:

- create deterministic small product DB fixture in tests or temp dir;
- include tenants, product_calls, manager mapping, capture inbox, job runs;
- no `stable_runtime` writes.

Tests:

- dashboard summary returns expected counters;
- capture table has stable `event_key` values;
- scheduler rows are sorted and shaped correctly.

### WP-3. Dashboard contract adapter

Scope:

- add a single backend aggregator for dashboard page data if existing endpoints
  are too granular;
- keep schema version explicit;
- include `actions.allowed` and `actions.blocked`.

Suggested route:

```text
GET /dashboard/appliance
```

Return shape:

```json
{
  "schema_version": "appliance_dashboard_v1",
  "summary": {},
  "capture": {},
  "scheduler": {},
  "writeback": {},
  "gates": {},
  "knowledge": {},
  "settings": {},
  "actions": {}
}
```

### WP-4. Frontend shell

Scope:

- build one local dashboard screen;
- dense operational UI, not marketing layout;
- no direct script execution from frontend;
- all data comes from Product API.

Expected controls:

- tabs for Dashboard / Capture / Scheduler / Writeback / Knowledge / Settings;
- status badges;
- tables;
- JSON/provenance drawer for selected event;
- refresh button.

### WP-5. Browser verification

Scope:

- run local dev/API server;
- open dashboard in browser;
- verify desktop and mobile widths;
- check empty/loading/error states;
- ensure no overlapping UI text.

## Safety acceptance criteria

The feature is accepted only if:

- no files under `stable_runtime` are modified by tests;
- no ASR/R+A code path is invoked;
- no AMO/Tallanto write method is invoked;
- UI has no live-write button;
- Product API remains read-only by default;
- every displayed row has a stable id/provenance field;
- tests cover read-only behavior and schema shape.

## Suggested implementation order

1. Add/verify Product API `GET /dashboard/appliance` aggregator.
2. Add temp product DB fixture tests.
3. Add HTTP route tests for dashboard, capture, scheduler and blocked mutation.
4. Build frontend dashboard shell against local API or fixture JSON.
5. Run browser QA.
6. Update runbook with launch instructions.

## Definition of done

Commands:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_product_api*.py tests/test_productization_*dashboard*.py
```

Manual smoke:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py --help
```

Expected outcome:

- dashboard opens locally;
- shows product DB/capture/scheduler/writeback readiness;
- works without runtime DB mutation;
- is demo-safe enough for internal use.

## Follow-up features after dashboard v1

1. Controlled capture runbook UI: show plan/dry-run/download reports without
   direct execution.
2. AMO writeback preview page: diff table, blockers, staged rollout checklist.
3. Knowledge Lab v1: sales signals, answer patterns, ROP validation queue.
4. Supervisor v1: job schedule, retry_wait/blocked explanations, health summary.
5. Client appliance setup wizard: tenant config, Mango credentials check, AMO
   readiness check, no secret values displayed.
