# Risk Review

## Главная защита

F2e не меняет runtime. Это report-only change.

## Что стало безопаснее

- Убрана старая логика `kb_partial` из строкового matcher-а.
- Payment/enrollment facts не могут стать доказательством, потому что proof идёт через F2d catalog.
- Enrollment-лексемы дополнительно закрыты на уровне catalog-фильтра.
- Grade parser ограничен порядковыми суффиксами и не принимает произвольные слова.
- Неполные запросы дают `needs_slot`/`unknown`, а не `kb_exact`.

## Риск следующего этапа

Следующий этап может захотеть понизить `manager_only`/`draft_for_manager` на 2 найденных кейсах. Это нельзя делать напрямую:

- нужен отдельный shadow/active подфлаг;
- route-only изменение должно быть строго ограничено `exists`/`not_offered` proof;
- текст не должен обещать live seats;
- P0/money/danger rows должны оставаться untouched.
