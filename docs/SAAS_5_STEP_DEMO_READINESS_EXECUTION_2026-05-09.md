# SaaS 5-Step Demo Readiness Execution

Дата: 2026-05-09
Ветка работ: SaaS/productization

## Scope

Выполнен последовательный проход по пяти шагам после 8-phase pre-processing
productization pass.

Ограничения соблюдены:

- `stable_runtime` DB/audio/transcripts не менялись;
- ASR/R+A не запускались;
- AMO/Tallanto/CRM writes не выполнялись;
- текущие batch/start/run-ui scripts не менялись.

## Results

| Step | Status | Result |
|---|---|---|
| 1. Read-only AMO/Tallanto snapshot export | Blocked by external APIs | AMO returned `401 Unauthorized`; Tallanto returned `HTTP 500` on read-only phone lookup. |
| 2. Rebuild sanitized real demo | Done | `297` product calls, `21` capture inbox items, `validation_ok=true`, `snapshots_written=0`. |
| 3. Dashboard demo-flow check | Done | Local dashboard/API served on `127.0.0.1:8766`; `/health`, `/dashboard/appliance`, and `/dashboard` responded. |
| 4. Diagnostics/service pack/playbook | Done | Diagnostics, service templates, tenant isolation scaffold, and demo playbook generated under product root. |
| 5. Processing acceptance gates | Expected block | `7/9` gates passed; blocked by `PROCESSING_QUALITY_EXTERNAL_READY`; warning: missing CRM/Tallanto snapshots. |

## Generated artifacts

```text
_local_archive_mango_api_downloads_20260507/sanitized_real_demo_appliance/sanitized_real_demo_report.json
_local_archive_mango_api_downloads_20260507/product_appliance/ops/diagnostics_20260509/diagnostics_manifest.json
_local_archive_mango_api_downloads_20260507/product_appliance/service_pack_20260509/service_pack_manifest.json
_local_archive_mango_api_downloads_20260507/product_appliance/tenant_isolation/tenant_isolation_20260509.json
_local_archive_mango_api_downloads_20260507/product_appliance/demo_pilot_playbook_20260509/demo_pilot_playbook.md
_local_archive_mango_api_downloads_20260507/product_appliance/processing_acceptance_gates/processing_acceptance_gates_20260509.json
```

## Snapshot blockers

AMO credentials from `prod_runtime_transfer/.env.private` were found, but the
read-only request returned `401 Unauthorized`. Token should be refreshed or
replaced before rerunning:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_amo_snapshot_export.py --help
```

Tallanto credentials from `mango_tallanto_transfer/.env.private` were found, but
the read-only phone lookup returned server-side `HTTP 500`. This looks like an
upstream Tallanto/API-side issue or an endpoint/method mismatch. Rerun after
checking credentials/path:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_tallanto_snapshot_export.py --help
```

## Next actions

1. Refresh AMO read-only token.
2. Verify Tallanto API endpoint/path with a small smoke test.
3. Rerun both snapshot exporters.
4. Rebuild sanitized real demo so `snapshots_written > 0`.
5. Rerun processing acceptance gates after the processing dialog provides
   explicit quality evidence.
