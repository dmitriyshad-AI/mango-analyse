# Implementation notes

## Проброс статуса

`project_bot_context()` добавляет безопасное поле:

```json
{"next_step_status": "active|needs_manager_review|empty"}
```

Полный `metadata.next_step` не отдаётся, потому что там может быть текст действия или ПДн.

`build_bot_safe_crm_context()` сохраняет это поле в `timeline_context.bot_context.items`.

`_direct_path_bot_safe_context_items()` переносит статус в direct-path prompt context.

## Prompt-инструкция

Инструкция добавляется только внутри уже включённого bot-safe блока.

Если в items есть `needs_manager_review` или `empty`, prompt получает правило:

- шаг не подтверждён;
- не утверждать его клиенту;
- предложить уточнить с менеджером;
- датированную историю подавать как прежние заметки с маркером свежести.

Если есть только `active`, хедж-фраза не добавляется.
