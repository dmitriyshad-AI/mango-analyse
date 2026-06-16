# TZ-123 — уточняющий вопрос вместо ухода к менеджеру

Дата: 2026-06-16  
Ветка: `codex/tz123-question-instead-of-handoff` от `main` `dd00d65e`

## Шаг 0: карта носителей

Проверены 3 существующих носителя:

1. `src/mango_mvp/channels/dialogue_contract_pipeline.py::_single_missing_slot_question`
   - Назначение: один вопрос, если не хватает ровно одного слота.
   - Статус: старый dialogue-contract pipeline; прямой путь pilot_gold_v1 его не вызывает.

2. `src/mango_mvp/channels/dialogue_contract_pipeline.py::_scope_clarification_question`
   - Назначение: вопрос про scope, например формат/класс.
   - Статус: старый pipeline за `TELEGRAM_Q_CLARIFY_SCOPE`; прямой путь pilot_gold_v1 его не вызывает.

3. `src/mango_mvp/channels/rules_engine.py` A2 qualify
   - Назначение: proactive qualify-вопросы.
   - Статус: legacy rules_engine; для пилота заморожен.

Решение: не плодить новый LLM/keyword-механизм и не трогать legacy-лазанью. Реализация сделана как post-layer прямого пути после `action_decision`, но с тем же контрактом single-missing/scope clarification: только whitelist слотов, только если слот реально разблокирует разные exact-факты, и только при `draft_for_manager`.

## Реализация

Флаг:

- `TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF`
- default OFF
- в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS` не добавлен

Основная точка:

- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py::apply_question_instead_of_handoff_layer`
- подключено в `src/mango_mvp/channels/subscription_llm_parts/provider.py` после `apply_deal_action_decision_layer`

Gate-in:

- только `route == draft_for_manager`;
- P0 не перебивается: проверка через `_deal_action_final_p0`, который использует `p0_recall_spec`/classifier/gate/model_p0;
- только `action_decision.action == answer_only`;
- слоты только `grade`, `subject`, `format`, `time`;
- вопрос задаётся только если слот реально меняет/разблокирует факты;
- учитываются known slots, `slot_provenance`, `do_not_reask_slots`, recent questions;
- после замены текста повторно вызывается `apply_authoritative_output_gate`;
- где gate не pass — откат к исходному результату.

## Тесты

Целевые NEG/POS:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -k 'tz123_question or question_instead_of_handoff' tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py
=> 13 passed, 584 deselected
```

Широкий затронутый набор:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py
=> 597 passed
```

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
=> 3301 passed, 5 skipped, 1 warning
```

## Микро-прогон OFF -> ON

Собран набор:

- `product_data/telegram_dynamic_test_sets/tz123_micro_20260616.jsonl`
- replay source: `product_data/telegram_dynamic_test_sets/tz123_micro_replay_source_20260616.jsonl`

Команды:

```text
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=0
python3 scripts/run_telegram_dynamic_client_sim.py --scenarios product_data/telegram_dynamic_test_sets/tz123_micro_20260616.jsonl --replay-from product_data/telegram_dynamic_test_sets/tz123_micro_replay_source_20260616.jsonl --out-dir runs/20260616_tz123_micro_OFF --parallel 4 --bot-mode codex --judge-mode fake --client-mode fake --memory-mode off --semantic-mode fake --semantic-verifier-mode fake --judge-prompt-version v9.1

TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=1
python3 scripts/run_telegram_dynamic_client_sim.py --scenarios product_data/telegram_dynamic_test_sets/tz123_micro_20260616.jsonl --replay-from product_data/telegram_dynamic_test_sets/tz123_micro_replay_source_20260616.jsonl --out-dir runs/20260616_tz123_micro_ON_r2 --parallel 4 --bot-mode codex --judge-mode fake --client-mode fake --memory-mode off --semantic-mode fake --semantic-verifier-mode fake --judge-prompt-version v9.1
```

Результат:

- OFF: `11 dialogs`, `13 turns`, `0 FAIL`, `0 hard_gate_failures`, `config_validity.invalid=false`.
- ON_r2: `11 dialogs`, `13 turns`, `0 FAIL`, `0 hard_gate_failures`, `config_validity.invalid=false`.
- ON_r2 `question_instead_of_handoff`: `fired=0`; `skipped.route_not_draft_for_manager=12`, `skipped_manager_action.send_crm_data=1`.

Вывод по микро-прогону: текущая direct-модель на этом микро-наборе уже отвечала `bot_answer_self_for_pilot`, поэтому новый слой не получил target-вход `draft_for_manager` и не должен был срабатывать. Это measurement limitation набора, не доказательство отсутствия эффекта на сырье с реальными draft_for_manager-уходами. Положительный fired-путь покрыт unit-тестами.

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- вопросы короткие, клиентские, без внутренних терминов;
- P0/жалоба/спор оплаты не перехватываются;
- CRM/баланс/менеджерские действия не перехватываются;
- бренд берётся из контекста, чужой бренд в вопросах не появляется;
- флаг выключен по умолчанию и не добавлен в боевой профиль.

Неблокирующий риск:

- динамический микро-набор на текущем профиле не поймал бывший `draft_for_manager`, поэтому прирост автономии нужно регрейдить на сырье, где реально есть уход из-за недостающего слота.

Manual gate для регрейда:

- любой вопрос на P0/жалобе = BLOCKED;
- любой повтор уже названного класса/предмета/формата = BLOCKED;
- `fired` должен появляться только на `draft_for_manager + answer_only + exact_facts`.
