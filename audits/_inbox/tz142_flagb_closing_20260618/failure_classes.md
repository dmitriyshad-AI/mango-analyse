# TZ-142 Failure Classes

## Класс 1: closing-artifact

Описание: клиент благодарит или завершает разговор, а бот продолжает продавать, вытаскивать продуктовые факты, просить телефон или обещать звонок.

Статус: исправлено для прямого пути.

Регрессии:

- `test_direct_path_applies_tone_close_detect_to_self_route_product_facts`
- `test_direct_path_tone_close_detect_replaces_cautious_handoff_without_phone_cta`
- динамический дым `closing_smoke_no_cta_v2`: 0 FAIL, `contact_requested=0`

## Класс 2: confirmed-camp-detail-question

Описание: вопрос клиента про лагерь содержит реальную потребность, например "нужен ли ноутбук", а исправление закрытия могло бы ошибочно вырезать подтвержденные детали.

Статус: покрыто отрицательным тестом.

Регрессия:

- `test_direct_path_tone_close_detect_does_not_cut_confirmed_camp_detail_question`

## Класс 3: flagB-p0-regression

Описание: флаг B меняет подбор фактов/широкий fallback так, что P0-вопрос про возврат начинает получать самостоятельное объяснение вместо безопасного менеджерского маршрута.

Статус: не исправлено в ТЗ-142, зафиксировано как blocker для включения B.

Доказательство:

- Большой ON-прогон: `p0_mishandled=1`
- Короткая ON-перепроверка после no-CTA фикса: `tz137_neg_p0_refund` остался `FAIL`

Рекомендация: отдельное ТЗ на P0-стабилизацию флага B.

## Класс 4: address-scope-overreach

Описание: на общий адресный вопрос УНПК бот может добавить лишний ЛВШ/лагерный адрес или связанные расписания.

Статус: не исправлялось в ТЗ-142.

Доказательство:

- `closing_smoke_no_cta_v2`, `tz142_neg_unpk_clean_closing_after_address`: `PASS_WITH_NOTES`, soft `wrong_scope` на первом ходе.

Рекомендация: отдельное ТЗ на scope-фильтр адресов и лагерных фактов.
