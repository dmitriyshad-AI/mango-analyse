# Quality Roadmap Block 2: Thread Memory

Дата: 2026-06-04

Ветка: `codex/quality-roadmap-block1-20260604`

База: `b72d5edd` (Block 1)

## Что реализовано

- Добавлен флаг `TELEGRAM_Q_THREAD_MEMORY`, default OFF.
- Существующая память `topic_focus` продолжает работать как раньше, чтобы не ломать принятую базу.
- При `TELEGRAM_Q_THREAD_MEMORY=1` deterministic augment может дополнить `topic_focus` из `dialogue_memory_view.known_slots`, если у слотов допустимый источник.
- Явный формат/класс из текущего вопроса побеждает старые слоты памяти: пример `а очно?` не оставляет `format=онлайн`.
- Дополненные тема/слоты попадают в `AnswerContract.current_question`, `subquestions[].needed_fact_keys`, `planner_slots` и `known_slots`.
- Augment не применяется при P0 и явной смене темы; active_brand остаётся только из канала.

## Основные точки кода

- `src/mango_mvp/channels/dialogue_contract_pipeline.py:39` — флаг `TELEGRAM_Q_THREAD_MEMORY`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:453` — чтение флага из context/env.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:745` — `_augment_contract_with_memory_topic`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:793` — `_memory_focus_for_contract`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:925` — explicit current turn value wins over memory.
- `tests/test_dialogue_contract_pipeline.py:5112` — начало Block 2 regression tests.

## Поведение по флагам

- `TELEGRAM_Q_THREAD_MEMORY` absent/0: known_slots-only память не продвигается в deterministic retrieval; старое поведение сохраняется.
- `TELEGRAM_Q_THREAD_MEMORY=1`: known_slots с допустимыми источниками могут восстановить тему эллипсиса перед retrieve.
- `TELEGRAM_Q_THREAD_MEMORY=1` + другие quality flags ON: invariant suite держит P0 и product-number safety.

## Что не менялось

- Новых LLM-вызовов нет.
- `build_understanding_prompt` не переписан.
- P0/pregate, brand logic, output gate, KB и stable_runtime не тронуты.
- Старый `topic_focus` baseline не выключался, потому он уже принят в базе.
