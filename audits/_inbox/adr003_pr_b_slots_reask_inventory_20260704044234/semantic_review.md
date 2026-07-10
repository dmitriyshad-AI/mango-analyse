# Semantic review

## Статус

`PASS_WITH_NOTES` для PR-B как default-OFF anti-reask механизма.

## Что проверено

PR-B не добавляет клиентских фактов и не меняет текст ответа. Он только помогает памяти не просить повторно уже выведенные моделью slot names (`grade`, `subject`, `format`), если эти hidden slots уже были записаны через `slots_gsf`.

## Смысловая безопасность

- Значения hidden slots не попадают в prompt или `known_slots`.
- Значения не становятся `client_confirmed_slots`.
- `SLOTS_REASK` без `slots_gsf` не создаёт hidden slots.
- OFF-поведение сохраняет старый `do_not_reask`.

## Остаточные notes

Этот блок не решает задачу `slots_gsf -> known_slots`. Старые regex G/S/F остаются до отдельного merge-решения с provenance `semantic_reading_llm` и запретом попадания в `client_confirmed_slots`.
