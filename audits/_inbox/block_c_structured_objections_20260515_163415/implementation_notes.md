# Implementation Notes

Дата: 2026-05-15
Блок: C
Статус: выполнено без AMO live-write

## Что сделано

Добавлен внутренний структурный слой возражений:

- `build_structured_objections()`;
- стабильный `objection_id`;
- сохранение `source_call_id`, даты, менеджера и позиции звонка;
- простая безопасная категория;
- JSON для preview/audit/export;
- флаг `objections_truncated`.

Человекочитаемое поле `AI-актуальные возражения` осталось строкой и строится из той же структуры.

## Что принципиально не сделано

Поле `AI-возражения структура` не добавлено в AMO payload и не считается AMO-полем. Оно должно появляться в AMO только после появления реального потребителя.

## Что не запускалось

- AMO live-write;
- rollback apply;
- ASR;
- Resolve+Analyze;
- Tallanto/CRM write;
- изменения `stable_runtime`.
