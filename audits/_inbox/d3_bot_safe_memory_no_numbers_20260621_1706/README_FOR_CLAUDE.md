# D3: запрет чисел/дат/расписания из памяти

Цель ревью: проверить доводку поверх `0d606c3`, чтобы bot-safe память не давала модели точные числа/даты/расписание как источник фактов.

Ветка: `codex/d3-phase01-botsafe-integration`

Ключевые файлы:

- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`
- `tests/test_bot_safe_direct_path_context.py`

Что проверить:

1. Prompt-инструкция запрещает числа/даты/проценты/цены/расписание/адреса из памяти как факты.
2. `_direct_path_bot_safe_memory_prompt_text()` скрывает точную конкретику из памяти.
3. Блок «Факты по вашему вопросу» не фильтруется.
4. Нить памяти сохраняется: «обсуждали расписание» остаётся.
5. Флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT` остался default OFF.
