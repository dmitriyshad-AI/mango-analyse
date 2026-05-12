# Customer Timeline Approved Context Pack Runbook

Дата: 2026-05-13

## Зачем нужен слой

Этот слой превращает валидное операторское решение `approve` в read-only пакет контекста для будущего бота или канального интерфейса.
Он не отправляет сообщения, не пишет в amoCRM/Tallanto, не меняет `customer_timeline.sqlite`, не трогает runtime-БД и `stable_runtime`.

## Что нужно на входе

Нужны три файла/источника:

- `customer_timeline.sqlite` - продуктовая база единой ленты клиента.
- `approval_workspace.json` - workspace из предыдущего этапа.
- `approval_decisions.filled.jsonl` - заполненный операторский JSONL из Stage 7.

Можно дополнительно передать `approval_decisions.validation.report.json`. Это audit/cache artifact.
Stage 8 не доверяет ему вслепую: он сам заново валидирует JSONL против workspace и сверяет cached report, если он передан.

Пакет создается только если:

- Stage 7 report имеет `validation_ok=true`.
- `workflow_status=approved_for_next_dry_run`.
- все строки решения фактически `approve`.
- workspace fingerprint совпадает с approval report.
- workspace имеет `status=ready_for_review`.
- у клиента нет открытых конфликтов в актуальной БД.
- есть хотя бы один `allowed_for_bot=true` и `requires_manager_review=false` chunk.

## Команда

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/customer_timeline_approved_context_pack.py \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --timeline-db "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline.sqlite" \
  --workspace-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_workspace.json" \
  --decisions-jsonl "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.filled.jsonl" \
  --approval-report-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approval_decisions.validation.report.json" \
  --out-pack-json "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/approved_context_pack.json"
```

Код возврата:

- `0` - approved context pack готов.
- `1` - входы прочитаны, но pack заблокирован бизнес-условием: reject, needs_rework, stale workspace, конфликт, нет safe context.
- `2` - ошибка путей, схемы JSON или запуска.

## Что внутри pack

- `source_refs` - SHA256 workspace/report и fingerprint, без абсолютных локальных путей.
- `decisions_jsonl_sha256` - хэш фактического операторского JSONL.
- `approval` - краткий снимок approved decision.
- `customer` - только безопасная карточка из read API с маскированными контактами.
- `approved_context.items` - только bot-safe chunks.
- `channel_context` - минимальный контекст, который можно передать в будущий draft builder.
- `safety` - явные флаги no-live/no-write.

## Safety gates

- `write_crm=false`
- `write_tallanto=false`
- `send_email=false`
- `send_messenger=false`
- `live_send=false`
- `run_asr=false`
- `run_ra=false`
- `write_runtime_db=false`
- `stable_runtime_writes=false`
- `network_calls=false`
- `subprocess_calls=false`
- `llm_calls=false`
- `rag_used=false`

## Следующий шаг

Следующий слой может подключить этот pack к channel preview service: входящее сообщение плюс approved context pack дают черновик ответа.
Даже после этого live-отправка должна оставаться выключенной до отдельного этапа approval/send gate.
