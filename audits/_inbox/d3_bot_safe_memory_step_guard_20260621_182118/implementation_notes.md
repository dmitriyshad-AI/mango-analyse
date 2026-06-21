# Implementation Notes

Цель: детерминированно не позволять боту утверждать конкретный шаг из безопасной памяти, если `next_step.status` равен `needs_manager_review` или `empty`.

Изменения:

- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py`
  - добавлен `apply_bot_safe_memory_step_guard`;
  - добавлен сбор `next_step_status` из метаданных direct path и из элементов `bot_safe_summary`;
  - добавлен узкий детектор конкретных шагов: бронь/место/запись/зачисление/гарантия/продвижение;
  - при срабатывании маршрут автономного ответа понижается до `draft_for_manager`, текст заменяется на безопасное уточнение.
- `src/mango_mvp/channels/subscription_llm_parts/provider.py`
  - гейт подключён после `apply_semantic_output_verifier` и до `apply_authoritative_output_gate`.
- `tests/test_bot_safe_memory_step_guard.py`
  - новые регрессионные тесты на review/empty/active/OFF/нейтральный хендофф.
- `tests/test_subscription_llm_draft_provider.py`
  - добавлен тест, что гейт срабатывает в прямом пути без включённого LLM-верификатора.

Флаг: `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, default OFF.
