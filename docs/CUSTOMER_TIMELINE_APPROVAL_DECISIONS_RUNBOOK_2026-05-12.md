# Customer Timeline Approval Decisions Runbook

Дата: 2026-05-12

## Зачем нужен слой

Этот слой фиксирует решение оператора по Customer Timeline Approval Workspace без live-действий.
Он не пишет в amoCRM, Tallanto, почту, мессенджеры, runtime-БД и `stable_runtime`.
Результат этапа - только JSONL-шаблон для оператора и JSON-отчет валидации.

## Что считается решением

Поддерживаются три финальных решения:

- `approve` - оператор проверил карточку и разрешает следующий безопасный dry-run.
- `reject` - оператор отклонил текущую карточку.
- `needs_rework` - оператор увидел проблему, которую нужно исправить до следующего шага.

Шаблон создается со статусом `pending`. Такой файл не считается валидным финальным решением.
Для финального решения обязательны `reviewer`, `reason` и timezone-aware `decided_at`.
Также обязательны:

- `reason_codes` из разрешенного списка: `reviewed_ok`, `identity_conflict`, `bot_context_issue`, `missing_context`, `data_quality_issue`, `operator_rejected`, `needs_rework`, `other`.
- `acknowledgements.understands_no_live_writes=true` для любого финального решения.
- все `acknowledgements=true` для `approve`.
- непустой `rework_items` для `needs_rework`.

Если workspace заблокирован конфликтом или в нем недостаточно контекста, `approve` запрещен. В таком случае допустимы только `reject` или `needs_rework`.

## Что есть в строке JSONL

Каждая строка самодостаточна и содержит:

- `decision_id` - стабильный идентификатор строки решения.
- `workspace_ref` - версия workspace, время генерации и SHA256 файла workspace, если шаблон создан из JSON.
- `workspace_summary_snapshot` - снимок статуса workspace на момент генерации.
- `queue_item` - пункт очереди проверки: конфликт, проверка bot context или ready-review.
- `decision`, `reviewer`, `reason`, `reason_codes`, `decided_at`.
- `acknowledgements` и `rework_items`.
- `safety` - явное подтверждение, что live-действий нет.

## Создать шаблон из готового workspace

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/customer_timeline_approval_decisions.py template \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --workspace-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_workspace.json" \
  --out-template-jsonl "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.template.jsonl" \
  --out-report-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.template.report.json"
```

## Создать шаблон напрямую из timeline DB

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/customer_timeline_approval_decisions.py template \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --timeline-db "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline.sqlite" \
  --tenant-id foton \
  --customer-id "<customer_id>" \
  --out-template-jsonl "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.template.jsonl" \
  --out-report-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.template.report.json"
```

## Проверить заполненный оператором JSONL

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/customer_timeline_approval_decisions.py validate \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --workspace-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_workspace.json" \
  --decisions-jsonl "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.filled.jsonl" \
  --out-report-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.validation.report.json"
```

Код возврата:

- `0` - шаблон создан или решения валидны.
- `1` - файл решений прочитан, но есть `pending`, пропуски, дубли, неизвестные строки или ошибки заполнения.
- `2` - ошибка запуска, пути или формата файла.

## Safety gates

- `write_crm=false`
- `write_tallanto=false`
- `send_email=false`
- `send_messenger=false`
- `run_asr=false`
- `run_ra=false`
- `write_runtime_db=false`
- `stable_runtime_writes=false`
- `network_calls=false`
- `subprocess_calls=false`

## Следующий шаг

После валидного решения `approve` этот слой разрешает только следующий read-only/dry-run пакет.
Он не является live-approval и не запускает запись в CRM.
