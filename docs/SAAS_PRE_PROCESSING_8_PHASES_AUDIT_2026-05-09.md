# SaaS Pre-Processing 8 Phases Audit

Дата: 2026-05-09
Ветка работ: SaaS/productization

## Scope

Этот проход закрывает productization-работы, которые можно выполнять до того,
как отдельный processing-диалог доведет качество обработки звонков до
достаточного уровня.

Запрещенные действия не выполнялись:

- не менялись `stable_runtime` DB/audio/transcripts;
- не запускались ASR/R+A;
- не выполнялись CRM/Tallanto writes;
- не менялись существующие batch/start/run-ui scripts.

## Phase results

| Phase | Result | Main artifacts |
|---|---|---|
| 1. CRM/Tallanto snapshots for real demo | Sanitized real demo reads JSON/JSONL/CSV snapshots and masks entities. | `sanitized_real_demo.py`, `test_productization_sanitized_real_demo.py` |
| 2. Demo-ready dashboard | Dashboard exposes `demo_readiness` panel and summary counters. | `saas_demo_contracts.py`, `product_api.py`, `product_api_http.py` |
| 3. Setup wizard | Wizard now emits config templates and install profile. | `appliance_config_wizard.py`, `mango_office_appliance_config_wizard.py` |
| 4. Ops hardening | Product ops now builds diagnostics bundle. | `product_ops.py`, `mango_office_product_ops.py` |
| 5. Service packaging | Service pack writes launchd/systemd templates only. | `appliance_service_pack.py`, `mango_office_appliance_service_pack.py` |
| 6. Multi-tenant isolation | Tenant report checks tenant-scoped rows and optional scaffold. | `tenant_isolation.py`, `mango_office_tenant_isolation.py` |
| 7. Demo/pilot playbook | Markdown/JSON playbook for client demo and pilot. | `demo_pilot_playbook.py`, `mango_office_demo_pilot_playbook.py` |
| 8. Processing acceptance gates | Read-only gates block processing integration until quality evidence is explicit. | `processing_acceptance_gates.py`, `mango_office_processing_acceptance_gates.py` |

## Safe command sequence

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_sanitized_real_demo.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_config_wizard.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_service_pack.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_tenant_isolation.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_demo_pilot_playbook.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_processing_acceptance_gates.py --help
```

## Verification

Focused tests were run after each phase. Final full verification:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `323 passed, 1 warning`.

Compile check also passed for all new/changed productization modules and CLI
entrypoints in this pass.

Additional audit:

- new CLI `--help` checks passed for service pack, tenant isolation,
  demo/pilot playbook and processing acceptance gates;
- `git diff --check` passed for all files touched in this pass.

## Self-audit

Findings and fixes during the pass:

- Phase 1: found risk of overlapping source/demo product roots and kept the
  guard that refuses this scenario.
- Phase 2: avoided adding new HTTP routes to keep existing Product API route
  contracts stable; demo readiness is exposed through `/dashboard/appliance`.
- Phase 3: config templates use placeholders only and do not write secrets.
- Phase 4: diagnostics bundle does not execute backup or restore.
- Phase 5: service pack does not install or start services.
- Phase 6: tenant scaffold is opt-in and never mutates product DB.
- Phase 7: playbook marks processing quality as an external blocker.
- Phase 8: acceptance gates are intentionally blocked until processing quality
  evidence is supplied explicitly.

## Remaining gaps

- Real CRM/Tallanto snapshots are still needed for a stronger external demo.
- Scheduler/service templates are generated, but not installed or tested as
  macOS launchd/systemd services.
- Processing acceptance cannot go green until the processing dialog produces
  explicit quality evidence.
