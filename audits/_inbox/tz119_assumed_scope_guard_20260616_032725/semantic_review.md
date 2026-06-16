# Semantic review

## Verdict

PASS_WITH_NOTES для инертного merge под default OFF.

## Audience

Клиент Telegram-бота и менеджер, который смотрит черновик.

## Что прошло

- Бот больше не должен уверенно утверждать класс/предмет/формат, если они пришли только из CRM/контекста без клиентской цитаты.
- При включённом стражe клиент получает мягкий уточняющий вопрос вместо цены/числа под неподтверждённый параметр.
- Подтверждённый ранее класс не ломается на коротких продолжениях диалога: если в памяти есть цитата клиента, слот считается подтверждённым.
- P0/рисковые обращения не переписываются этим стражем.
- Маршрут не повышается к менеджеру, поэтому цель "0 новых уходов" соблюдена на уровне кода.

## Неблокирующие риски

- Страж ловит явные утверждения класса/предмета/формата/продукта в тексте. Если модель назовёт число без явного параметра, финальную проверку по числам продолжает держать существующий authoritative output gate.
- Ночной замер OFF->ON на real_002/012 и остальных real кейсах не запускался в этом проходе, потому что это отдельный тяжёлый прогон.
- Текст уточнения универсальный; после пилотного замера может потребоваться сделать его теплее и ближе к стилю менеджеров.

## Регрессионные проверки

- `test_tz119_model_driven_requires_assumed_scope_guard`
- `test_tz119_unconfirmed_crm_grade_is_soft_scope_not_hard_demotion`
- `test_tz119_confirmed_grade_still_scope_demotes_wrong_fact`
- `test_tz119_assumed_scope_guard_reasks_without_manager_handoff`
- `test_tz119_confirmed_slot_quote_prevents_reask_on_ellipsis`
- `test_tz119_assumed_scope_guard_skips_p0_risk`
- `test_tz119_draft_prompt_marks_assumed_slots_only_when_flag_enabled`

## Recommended next action

Регрейд Claude #1 по сырью, затем ночной замер флага ON на real_002/012 и контроль "unsupported-param падает, over-handoff/reask не растут".
