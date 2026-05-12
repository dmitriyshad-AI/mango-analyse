# SaaS Productization Stages 1-4 Audit

Дата: 2026-05-09

Ветка работ: SaaS/productization

## Scope

Проход закрыл первые четыре глобальных этапа, не затрагивая обработку звонков:

1. Client-hosted appliance.
2. Dashboard as product interface.
3. CRM/Tallanto read-only readiness.
4. Supervisor/scheduler readiness.

## Safety

- `stable_runtime` DB/audio/transcripts не менялись.
- ASR/R+A не запускались.
- AMO/Tallanto/CRM live write не выполнялся.
- Runtime DB не менялась.
- Новые записи ограничены product root reports/snapshots и тестовыми temp dirs.

## Stage 1. Client-hosted appliance

Добавлено:

- `src/mango_mvp/productization/appliance_command_surface.py`
- `scripts/mango_office_appliance.py`
- `tests/test_productization_appliance_command_surface.py`

Результат: оператор получает единый command-surface report с безопасным порядком
действий: demo/bootstrap, config wizard, healthcheck, scheduler health,
dashboard serve, backup/restore dry-run.

Аудит: command surface не исполняет команды, а только печатает/пишет report.
Path guards запрещают `stable_runtime` и output вне product root.

## Stage 2. Dashboard product UI

Добавлено:

- фильтры Product API для `capture_recent`;
- фильтры Product API для `scheduler_runs`;
- query params для `/dashboard/appliance`, `/capture/recent`, `/scheduler/runs`;
- UI controls в `/dashboard`: search, manager, capture status, job status, job type.

Аудит: фильтры параметризованы, SQL не строится из user text напрямую, длина
filter values ограничена. UI не добавляет action buttons для ASR/R+A/CRM write.

## Stage 3. CRM/Tallanto read-only mapping preview

Добавлено:

- `src/mango_mvp/productization/crm_tallanto_mapping_preview.py`
- `scripts/mango_office_crm_tallanto_mapping_preview.py`
- `tests/test_productization_crm_tallanto_mapping_preview.py`
- Product API endpoint `GET /crm/mapping-preview`

Результат: product layer может сверить capture rows с локальными AMO/Tallanto
snapshots и показать `resolved`, `missing`, `ambiguous` без live CRM calls.

Аудит: Tallanto остается read-only context provider. Preview не пишет в CRM,
не делает сетевые calls, не снимает writeback gates.

## Stage 4. Supervisor/scheduler readiness

Добавлено:

- `src/mango_mvp/productization/scheduler_control_plane.py`
- `scripts/mango_office_scheduler_control_plane.py`
- `tests/test_productization_scheduler_control_plane.py`
- Product API endpoint `GET /scheduler/control-plane`

Результат: scheduler panel теперь показывает не только health, но и recommended
actions: schedule shadow poll, run scheduler tick, review failed jobs, requeue
stale locks.

Аудит: control-plane report не исполняет jobs. Он только строит команду, которую
оператор может выполнить отдельно.

## Tests

Focused tests:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_appliance_command_surface.py \
  tests/test_productization_appliance_dashboard.py \
  tests/test_productization_crm_tallanto_mapping_preview.py \
  tests/test_productization_scheduler_control_plane.py \
  tests/test_productization_product_api.py \
  tests/test_productization_product_api_http.py
```

Result: `25 passed`.

Full productization suite:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `300 passed, 1 warning`.

Compile and diff checks:

```zsh
PYTHONPYCACHEPREFIX=/private/tmp/mango_pycache_saas_1_4 PYTHONPATH=src python3 -m py_compile ...
git diff --check -- <changed SaaS/productization files>
```

Result: passed.

## Residual limitations

- Dashboard browser screenshot QA may still be blocked by local Browser security
  policy for `127.0.0.1`; HTTP/API smoke remains available.
- CRM/Tallanto preview depends on local snapshots. Live Tallanto snapshot export
  remains a separate guarded/read-only CRM task.
- Scheduler control-plane does not replace a real service manager; it is the safe
  local readiness/control report before packaging.
