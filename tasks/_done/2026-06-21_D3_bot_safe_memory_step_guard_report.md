# D3: детерминированный гейт спорного шага из памяти

Дата: 2026-06-21
Ветка: `codex/d3-phase01-botsafe-integration`

## Что сделано

- Добавлен отдельный детерминированный гейт выхода `apply_bot_safe_memory_step_guard`.
- Гейт работает только при включённом `TELEGRAM_BOT_SAFE_CRM_CONTEXT`.
- Статус `next_step.status` берётся из метаданных прямого пути или напрямую из элементов безопасной памяти `bot_safe_summary`.
- Если статус памяти `needs_manager_review` или `empty`, а черновик утверждает конкретный спорный шаг, текст заменяется на безопасное уточнение: `Уточню актуальный шаг с менеджером и вернусь с ответом.`
- Гейт подключён после смыслового проверяющего слоя и до итогового выходного гейта, поэтому не зависит от доступности LLM-верификатора.

## Что не входит

- Не менял автономию и право самостоятельной отправки.
- Не менял self-route политику для памяти review/empty.
- Не трогал AMO, Tallanto, live-бота, боевую БД и stable_runtime.

## NEG-проверки

- `active`-память не режется.
- Нейтральный хендофф `Передам менеджеру, он свяжется...` не режется.
- `Менеджер свяжется завтра` не попадает в новый гейт, чтобы не дублировать существующий `FOLLOWUP_DEADLINE_RE`.
- При выключенном `TELEGRAM_BOT_SAFE_CRM_CONTEXT` поведение не меняется.
- Гейт срабатывает в прямом пути даже без включённого смыслового LLM-верификатора.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_bot_safe_memory_step_guard.py tests/test_subscription_llm_draft_provider.py::test_direct_path_bot_safe_memory_step_guard_runs_without_semantic_verifier`
  - `7 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py`
  - `485 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3524 passed, 5 skipped, 1 warning`

## Остаточный риск

- Реальные кейсы 04/13/18 не прогонялись тяжелым M1-замером в этом шаге. Кодовый слой закрыт тестами на механизм; поведенческий регрейд по сырью остаётся для следующего переизмерения D7.
