# Runtime artifacts current index

Дата: 2026-05-11
Назначение: единая карта текущих и устаревших runtime-артефактов без удаления файлов.

## Current production layer

| Класс | Актуальный путь | Статус |
|---|---|---|
| Current runtime contract | `stable_runtime/CURRENT_RUNTIME.json` | current |
| Active AMO export pointer | `stable_runtime/CANONICAL_EXPORT.txt` | указывает на strict post-backfill export |
| Canonical DB | `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db` | current |
| Strict AMO export | `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_ru.csv` | current |
| Stage15 transcript gate | `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json` | current |
| CRM writeback quality gate | `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/summary.json` | current |
| AMO writeback queue | `stable_runtime/amo_writeback_queue_20260510_v2_production/summary.json` | current |
| Operator status | `stable_runtime/operator_status_20260511_v4_waiting_work/operator_status.json` | current after this pass |
| Runtime artifact index | `stable_runtime/runtime_artifact_index_20260511_v1/artifact_index.json` | current read-only index |

Current index counters:

```text
entries=422
active_current=8
blocked=14
audit_only=71
legacy_candidates=39
invalid_json_artifacts=1
```

## Current waiting/blocked work

| Класс | Путь | Что означает |
|---|---|---|
| Duplicate staff tasks | `stable_runtime/amo_duplicate_staff_tasks_20260511_v1/` | сотрудники объединяют дубли |
| After-staff recheck | `stable_runtime/amo_duplicate_after_staff_done_20260511_v1/` | пока 0 candidates, 13 blocked |
| Waiting autonomous work | `stable_runtime/amo_waiting_autonomous_work_20260511_v1/` | 1 non-duplicate, 40 refresh, 15 readback, 1 mismatch |
| Stage rollout blocked | `stable_runtime/amo_stage50_stage86_preflight_blocked_20260511_v1/` | live stage blocked until prerequisites |

## Audit-only artifacts

Папки `audits/_inbox/*` и `audits/_results/*` являются формальными артефактами обмена с Claude/GPT-аудитом. Их нельзя удалять до закрытия соответствующего этапа. Старые audit packs можно переносить в архив только после отдельного cleanup manifest.

## Legacy / superseded candidates

Кандидаты на перенос в quarantine уже перечислены в:

- `stable_runtime/project_cleanup_manifest_20260511_v1/manifest.csv`
- `stable_runtime/project_cleanup_manifest_20260511_v1/summary.json`

Текущий проход не удаляет и не перемещает файлы. До фактической очистки нужно вручную проверить строки `requires_human_review`.

## Правило на будущее

Любой новый production-путь должен быть отражен в `CURRENT_RUNTIME.json` или в operator status. Любой старый путь, который не входит в current production layer, должен быть либо audit-only, либо candidate-to-quarantine с restore path.
