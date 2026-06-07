# TRACE hp_topic_change — compound question not decomposed

Дата: 2026-06-07

Scope: read-only trace по локально доступному сырью. Пинованный `dynamic_dialog_transcripts.jsonl` с `hp_topic_change` в локальных `runs/` и `audits/_inbox/` не найден, поэтому field-level trace `bot_dialogue_contract_pipeline` пока не подтверждён. Использованы:

- `D1_audit_backlog/READING_PACK_for_Dmitry_2026-06-06.md`
- `D1_audit_backlog/TZ_tone_block_2026-06-06.md`
- `D1_audit_backlog/REGRADE_step4_four_runs_2026-06-06.md`
- `D1_audit_backlog/REGRADE_verifier_four_runs_2026-06-06.md`

## Наблюдение по сырью

T1 клиент: спрашивает сразу две темы: годовой онлайн-курс математики для 9 класса и летний лагерь.

T1 бот: отвечает только по лагерю, онлайн-математику не покрывает.

T2 клиент: уточняет, что лагерь — отдельная программа, и спрашивает про занятия/возраст.

T2 бот: смешивает разные лагерные продукты: ЛВШ Менделеево и городской лагерь Фотона в Москве.

T3 бот уже исправляется на городской летний лагерь в Москве.

## Механизм

1. Ошибка T1 похожа не на тон и не на финальный санитайзер, а на потерю второй части в понимании/контракте. Если `AnswerContract` содержит только `camp`, `_partial_yield_result_before_handoff` не может восстановить отдельную часть `online math 9 annual course`, потому что helper работает по уже разложенным частям.

2. Ошибка T2 — scope продукта: `camp_lvsh` и городской летний лагерь Фотона должны быть разными продуктами. Даже если память удержала camp-контекст, retrieval/композиция не должны подмешивать факты ЛВШ Менделеево в ответ про городской лагерь.

3. Верификатор позже ловит часть проблемы как `derived_product_claim|contradicts`, но это поздний предохранитель. Корень — contract decomposition + product scope.

## План фикса

1. В `build_understanding_prompt`/парсинге `AnswerContract` явно требовать compound decomposition: если клиент спрашивает два продукта/темы в одном ходе, контракт должен хранить обе части, а не выбирать одну.

2. Прокинуть части в partial-yield: каждая часть получает свой `product_scope`/`fact_keys`/`missing_fact_keys`, затем ответ строится как "покрытая часть + честно про непокрытую".

3. Добавить scope guard для лагерей: `city_summer_camp` != `lvsh_mendeleevo`. Факт другого лагеря может быть `adjacent`, но не grounding для ответа.

4. Регрессии:
   - `hp_topic_change` T1: ответ покрывает онлайн-математику 9 класса и лагерь либо честно отдаёт непокрытую часть менеджеру.
   - `hp_topic_change` T2: ответ про городской лагерь не содержит фактов ЛВШ Менделеево.
   - NEG: одинарный вопрос про ЛВШ Менделеево не режется city-camp guard'ом.
   - NEG: P0 в любой части составного хода переводит весь ход к менеджеру.

## Что нужно для точной трассы

Нужен исходный JSONL пинованного прогона с `hp_topic_change`, чтобы подтвердить:

- `contract.current_question`
- `contract.needed_fact_keys`
- `contract.missing_fact_keys`
- `bot_dialogue_contract_pipeline.retrieved_fact_keys`
- `handoff_trace`/`partial_yield` metadata

Без этого вывод выше является механизмной гипотезой по сырому тексту диалога и регрейдам, но не field-level доказательством.
