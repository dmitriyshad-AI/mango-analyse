# SaaS Phases 1-3 Completion Report

Дата: 2026-05-09

Scope: SaaS/productization ветка. Processing/runtime DB/audio/transcripts не
менялись, ASR/R+A не запускались, AMO/Tallanto live writes не выполнялись.

## Phase 1. Baseline hardening and audit intake

Status: complete for current productization baseline.

Что закрыто:

- принят и разложен project risk audit;
- создан response-plan;
- P0/P1 live writeback risks закрыты explicit confirmation gates;
- legacy `sync_amocrm` оставлен только как disabled maintenance path;
- scripts safety matrix и full scripts catalog созданы;
- README/Makefile/runbook больше не скрывают dangerous side effects;
- AMO/Tallanto field mapping зафиксирован;
- current architecture и data model зафиксированы.

Key docs:

- `docs/PROJECT_RISK_AUDIT_RESPONSE_PLAN_2026-05-09.md`
- `docs/SCRIPT_SAFETY_MATRIX.md`
- `docs/CLI_AND_SCRIPTS_CATALOG_2026-05-07.md`
- `docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md`
- `docs/ARCHITECTURE_CURRENT.md`
- `docs/DATA_MODEL.md`

Key tests:

- `tests/test_amo_writeback_guards.py`
- `tests/test_legacy_amocrm_sync_guard.py`

## Phase 2. Supervised local Product API service

Status: complete for read-only local appliance v1.

Что закрыто:

- Product API facade работает как read-only contract layer;
- HTTP layer имеет health/readiness и mutation blocking;
- добавлен единый dashboard aggregator:
  `GET /dashboard/appliance`;
- route-level JSON errors сохраняются для invalid query params;
- readiness report проверяет все read-only routes;
- Product API service запускается локально через existing script.

Key code:

- `src/mango_mvp/productization/product_api.py`
- `src/mango_mvp/productization/product_api_http.py`
- `scripts/mango_office_product_api_http.py`

Key routes:

- `GET /health`
- `GET /dashboard/appliance`
- `GET /dashboard/summary`
- `GET /capture/recent`
- `GET /queues/processing`
- `GET /scheduler/runs`
- `GET /asr/gates`
- `GET /writeback/previews`
- `GET /knowledge/playbook`
- `GET /settings/adapters`

Safety:

- only `GET` routes are allowed;
- `POST`, `PUT`, `PATCH`, `DELETE` return blocked mutation responses;
- dashboard actions keep `run_asr=false`, `run_ra=false`, `write_crm=false`,
  `write_runtime_db=false`.

## Phase 3. UI v1 over Product API

Status: complete as a first read-only operational shell.

Что закрыто:

- added local dashboard HTML shell served from Product API HTTP layer;
- dashboard reads data only from `GET /dashboard/appliance`;
- visible panels:
  - Summary metrics;
  - Capture inbox;
  - Scheduler;
  - Writeback readiness;
  - Knowledge readiness;
  - Settings/adapters;
  - Safety gates;
  - Selected JSON payload;
- UI has no direct script execution and no live-write button.

Route:

```text
GET /dashboard
```

The root route also opens the dashboard:

```text
GET /
```

Manual launch:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py \
  --product-root _local_archive_mango_api_downloads_20260507/product_appliance \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  serve --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/dashboard
```

## Verification commands

```zsh
PYTHONPYCACHEPREFIX=/private/tmp/mango_pycache_dashboard PYTHONPATH=src python3 -m py_compile \
  src/mango_mvp/productization/product_api.py \
  src/mango_mvp/productization/product_api_http.py \
  tests/test_productization_appliance_dashboard.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_product_api.py \
  tests/test_productization_product_api_http.py \
  tests/test_productization_appliance_dashboard.py
```

## Remaining gaps before Phase 4

- Browser visual QA should be repeated after committing/stabilizing the current
  branch state.
- UI v1 is intentionally operational and read-only; it is not yet client-demo
  polished.
- Product API still uses SQLite appliance DB. This is expected for current
  client-hosted/local phase.
- Processing bridge remains dry-run only and belongs to later phases.

## Next phase

Proceed to Phase 4: Mango capture from shadow to controlled ingest.
