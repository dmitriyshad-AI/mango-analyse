# SaaS Appliance Hardening Pack Audit

Дата: 2026-05-09

Ветка работ: SaaS/productization

## Scope

Этот проход закрывает шесть безопасных productization-шагов:

1. read-only AMO snapshot exporter;
2. anonymized demo tenant;
3. dashboard polish;
4. appliance config wizard;
5. backup/restore/ops hardening;
6. scheduler health/readiness.

## Safety contract

- `stable_runtime` DB/audio/transcripts не меняются.
- ASR/R+A не запускаются.
- AMO/Tallanto/CRM live write не выполняется.
- Runtime DB не меняется.
- Все новые write effects ограничены product root: reports, demo DB, CRM snapshot,
  backup files, wizard/healthcheck reports.

## Implementation summary

| Area | Artifact | Result |
|---|---|---|
| AMO snapshot | `src/mango_mvp/productization/amo_snapshot_exporter.py` | Читает amoCRM contacts/leads и пишет `crm_snapshots/amocrm_entities.json`. При linked leads не создает лишний duplicate contact candidate по тому же телефону. |
| Demo tenant | `src/mango_mvp/productization/demo_tenant.py` | Создает fake product root: SQLite product DB, CRM snapshot, fake calls, capture inbox, scheduler rows, demo reports. |
| Dashboard | `src/mango_mvp/productization/product_api.py`, `product_api_http.py` | Добавлены `/scheduler/health`, `/processing/lifecycle`, панели Lifecycle/CRM Preview/Safety, scheduler due/failed metrics. |
| Wizard | `src/mango_mvp/productization/appliance_config_wizard.py` | Проверяет paths, DB, Mango env, CRM snapshot, retention, backup dir и runtime separation. |
| Ops | `src/mango_mvp/productization/product_ops.py` | Healthcheck, product DB backup, backup verify, restore dry-run. SQLite quick_check работает с путями с пробелами. |
| Scheduler health | `src/mango_mvp/productization/scheduler_health.py` | Отчет по due, failed, blocked, running, stale running и locked jobs. |

## Commands

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_amo_snapshot_export.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_demo_tenant.py --replace
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_config_wizard.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py healthcheck
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py backup
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_scheduler_health.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_api_http.py serve --host 127.0.0.1 --port 8765
```

## Audit findings and fixes

| Finding | Risk | Fix | Status |
|---|---|---|---|
| AMO exporter could emit both contact and linked lead for the same phone. | CRM resolver would block normal linked-lead cases as ambiguous. | Prefer linked lead candidates; emit standalone contact only when contact has no linked leads. | Fixed |
| SQLite quick_check used an unquoted URI path. | Product root paths with spaces could fail backup verification. | Quote SQLite file URI with `urllib.parse.quote`. | Fixed |
| Dashboard metric grid had a fixed six-column layout while seven metrics were rendered. | Visual wrapping could be awkward on desktop. | Switched metrics grid to responsive `auto-fit`. | Fixed |
| New scripts were not visible in safety/catalog docs. | Operator could treat them as `REVIEW_REQUIRED`. | Added entries to CLI catalog and safety matrix. | Fixed |

## Residual limitations

- AMO snapshot exporter still depends on valid amoCRM OAuth/token configuration and does not refresh tokens.
- Demo tenant is intentionally synthetic; it validates product flow, not transcript quality.
- Dashboard HTML shell has unit coverage, but no browser screenshot QA was run in this pass.
- Backup command copies the SQLite DB file and verifies it; a live multi-writer mode would need a stricter backup protocol or maintenance window.

## Verification plan

Run:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_amo_snapshot_exporter.py \
  tests/test_productization_demo_tenant.py \
  tests/test_productization_appliance_config_wizard.py \
  tests/test_productization_product_ops.py \
  tests/test_productization_scheduler_health.py \
  tests/test_productization_product_api.py \
  tests/test_productization_product_api_http.py \
  tests/test_productization_appliance_dashboard.py
```

Then run full productization tests:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

## Verification results

2026-05-09:

- Focused hardening tests: `29 passed, 1 warning`.
- Full productization suite: `289 passed, 1 warning`.
- Compile check with `PYTHONPYCACHEPREFIX=/private/tmp/mango_pycache_saas_hardening`: passed.
- Targeted whitespace/diff check for changed files: passed.
- Demo Product API smoke on `/private/tmp/mango_appliance_dashboard_qa`: `/dashboard/appliance`
  returned demo capture rows and read-only safety flags.
- Browser screenshot QA was attempted through the in-app Browser plugin, but local
  navigation to `http://127.0.0.1:8766/dashboard` was blocked by browser security
  policy. No browser fallback was used.
