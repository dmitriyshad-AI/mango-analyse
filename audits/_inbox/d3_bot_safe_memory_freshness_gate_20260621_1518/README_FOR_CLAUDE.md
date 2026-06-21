# D3: bot-safe memory freshness gate

Цель ревью: проверить, что `next_step.status` безопасно дошёл до prompt-блока прямого пути, а active-кейсы не получили лишний хедж.

Ветка: `codex/d3-phase01-botsafe-integration`

Флаг: `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, default OFF.

Ключевые файлы:

- `src/mango_mvp/customer_timeline/read_api.py`
- `src/mango_mvp/customer_timeline/bot_safe_runtime_context.py`
- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`
- `tests/test_bot_safe_runtime_context.py`
- `tests/test_bot_safe_direct_path_context.py`
- `tests/test_customer_timeline_read_api.py`
- `tests/test_telegram_dynamic_client_sim.py`
- `tests/test_run_amo_wappi_draft_loop.py`

Что проверить:

1. В read_api и runtime builder отдаётся только `next_step_status`, без текста спорного шага.
2. Prompt block:
   - `active` -> продолжить нить без лишних оговорок;
   - `needs_manager_review` / `empty` -> не утверждать шаг, использовать маркер свежести для датированной истории.
3. Active-only prompt не содержит фразу `по прежним заметкам, актуальность уточню`.
4. Измерительный контур получает `next_step_status`.
5. Флаг остался default OFF; live/AMO/Tallanto/боевая БД не трогались.
