# Quality Roadmap Block 1: Partial Yield

Дата: 2026-06-04

Ветка: `codex/quality-roadmap-block1-20260604`

База: `aae2e3b0`

## Что реализовано

- Добавлен флаг `TELEGRAM_Q_PARTIAL_YIELD`, default OFF.
- Добавлен общий helper partial-yield в `dialogue_contract_pipeline`: если перед уходом к менеджеру есть проверенный client-safe факт по части вопроса и есть missing-подвопрос, бот может ответить грунтованную часть и честно передать missing менеджеру.
- Helper вставлен в безопасные точки перед handoff через общий recover-путь и ветку `estimate_guard_failed`.
- Travel/route estimate разрешается до раннего ухода только при двух флагах: `TELEGRAM_Q_PARTIAL_YIELD=1` и `TELEGRAM_A_FREE_NUMBER_GATE=1`.
- Partial-yield candidate проходит full-text `verify_output`, operational checks и `check_claim_faithfulness`; `_hard_check` здесь намеренно не используется как единственная проверка, потому что он может вырезать handoff-часть и не проверить factual-префикс целиком.
- P0/refund/high-risk не повышаются и не подменяются частичным фактом.
- В metadata боевого результата добавлены `partial_yield_applied`, `partial_yield_fact_keys`, `partial_yield_missing`.

## Основные точки кода

- `src/mango_mvp/channels/dialogue_contract_pipeline.py:38` — флаг `TELEGRAM_Q_PARTIAL_YIELD`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:446` — чтение флага из context/env.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:1146` — travel/route estimate override перед ранним уходом.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:1986` — применение travel override к `estimate_policy`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:2096` — partial-yield перед fallback в `estimate_guard_failed`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:3675` — общий `_partial_yield_result_before_handoff`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:3762` — full-text verification partial-yield candidate.
- `src/mango_mvp/channels/subscription_llm.py:1918` — metadata для регрейда.
- `src/mango_mvp/channels/subscription_llm.py:8329` — safety flag `dialogue_contract_partial_yield_applied`.

## Поведение по флагам

- `TELEGRAM_Q_PARTIAL_YIELD` отсутствует или `0`: старое поведение сохраняется.
- `TELEGRAM_Q_PARTIAL_YIELD=1`: partial-yield может заменить handoff только при grounded fact + missing + чистых проверках.
- `TELEGRAM_Q_PARTIAL_YIELD=1` + `TELEGRAM_A_FREE_NUMBER_GATE=1`: travel/logistics estimate может дойти до estimate-composer, если вопрос не продуктовый и фактов нет.

## Отклонения от простого плана

Первичная реализация через `_hard_check` была недостаточной: негативный тест с foreign-brand fact показал, что handoff parser может не проверить factual-префикс candidate целиком. Поэтому helper использует отдельный full-text check поверх `verify_output` + semantic faithfulness.
