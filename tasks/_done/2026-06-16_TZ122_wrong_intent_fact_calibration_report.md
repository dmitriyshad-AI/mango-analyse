# TZ-122 wrong_intent_fact calibration report

Дата: 2026-06-16

## Что проверено по шагу 0

Подтверждено: в живом direct-path `wrong_intent_fact` рождается не в модели и не в списке `GATE_BLOCKING_CODES`.

- `src/mango_mvp/channels/subscription_llm_parts/provider.py:1034-1143` строит direct-path draft и затем вызывает `apply_authoritative_output_gate`.
- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:3225-3252` строит контракт и вызывает `verify_dialogue_contract_output`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:4073-4130` внутри `verify_output` вызывает `_wrong_intent_fact_findings`.
- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:832-872` не содержит `wrong_intent_fact` в `GATE_BLOCKING_CODES`, значит сам финальный gate не делает его блокирующим кодом.
- Реальное понижение возникает через fail/recover ветки вокруг cite-only recovery: `dialogue_contract_pipeline.py:5757-5825`, recoverable-коды `5848-5861`, hard P0/refund блокировка `5869-5882`.

## Что изменено

Флаг:

- `TELEGRAM_WRONG_INTENT_FACT_CALIBRATION`, default OFF.

Изменения под флагом:

1. Адресная ветка:
   - расширено распознавание адресного вопроса для формулировок вида `где/куда ... курсы/занятия/приходить/пробное`;
   - если адресный факт покрывает адресный `needed_fact_key` и scope, `wrong_intent_fact` не выставляется.

2. Лагерная ветка:
   - при ON демоут остаётся только для рассогласования scope лагеря: например, клиент спрашивает городской лагерь без проживания, а ответ использует ЛВШ с проживанием;
   - scope сравнивается только по лагерному факту, который реально виден в черновике, а не по соседнему факту из fact pack;
   - случай `сколько стоит смена`, где контракт потерял слово `лагерь`, больше не маскируется этим правилом. Это остаётся задачей C2 по ретриверу/контракту.

3. Контактные часы:
   - правило не ослаблено: график офиса/контактные часы по-прежнему нельзя выдавать как расписание занятий.

Не трогалось:

- `derived_product_claim`;
- `fact_grounding`;
- `unsupported_promise`;
- `unsupported_entity`;
- числовой scope guard;
- P0/refund;
- brand guard.

## Микрозамер правила

Файл вывода: `runs/tz122_wrong_intent_fact_20260616/rule_micro_measurement.txt`.

OFF:

- total `wrong_intent_fact`: 5
- address: 3
- camp: 1
- contact_hours: 1

ON:

- total `wrong_intent_fact`: 3
- address: 1
- camp: 1
- contact_hours: 1

Смысл результата:

- POS `где очные курсы`, `куда приходить на пробное`, `сколько стоит смена` перестали получать ложный `wrong_intent_fact`;
- POS с городским лагерем и соседним ЛВШ-фактом в fact pack не получает ложный scope-mismatch;
- NEG `цена -> адрес`, `контактные часы -> расписание занятий`, `городской лагерь -> ЛВШ с проживанием` остались под демоутом;
- старый OFF не ловил лагерный scope mismatch, ON ловит.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz122_wrong_intent_fact.py
8 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz122_wrong_intent_fact.py tests/test_subscription_llm_draft_provider.py -k 'wrong_intent or address_fact or contact_address or authoritative_output_gate or assumed_scope'
22 passed, 458 deselected

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_tz122_wrong_intent_fact.py
480 passed in 7.77s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz122_wrong_intent_fact.py tests/test_telegram_dynamic_client_sim.py
108 passed in 1.08s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3286 passed, 2 skipped, 1 warning in 47.92s
```

Предупреждение: старое `urllib3`/`LibreSSL`, не связано с ТЗ-122.

## Semantic review

Вердикт: `PASS_WITH_NOTES`.

Что прошло:

- бот перестаёт уходить к менеджеру на адресном ядре, если клиент реально спрашивает `где/куда`;
- бот перестаёт считать лагерный факт ошибкой только из-за потерянного слова `лагерь` в контракте;
- реальные опасные случаи не ослаблены: контактные часы как расписание, чужой лагерный scope, адрес вместо цены.

Остаточный риск:

- полный динамический прогон с моделью не запускался в этой задаче; проверен детерминированный слой, где рождается `wrong_intent_fact`;
- лагерная ветка всё ещё требует C2-замера ретривера: почему контракт/ретривер теряет лагерный intent в части кейсов.

Рекомендация:

- перед включением флага в пилотном профиле дать Claude #1 регрейд по сырью;
- после регрейда включать только вместе с микросетом C0/C2 и мониторить рост `derived_product_claim`, `fact_grounding`, `unsupported_entity`.
