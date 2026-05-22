# Telegram AI Employee Funnel v1 Implementation Notes

Дата: 2026-05-23

## Что реализовано

- Добавлен детерминированный слой воронки нового лида: `src/mango_mvp/channels/new_lead_funnel.py`.
- Добавлена внутренняя менеджерская сводка: `src/mango_mvp/channels/manager_handoff_summary.py`.
- Telegram public pilot runtime теперь:
  - собирает `funnel_state` до LLM-вызова;
  - передаёт в prompt `known_slots`, `missing_slots`, `next_best_question`, `next_step_type`;
  - пересобирает `funnel_state` после LLM-результата;
  - сохраняет `funnel_state`, `lead_stage`, `client_segment`, `next_step_type`, `known_slots`, `missing_slots`, `semantic_flags` и `manager_summary` в draft metadata.
- Prompt builder теперь явно объясняет модели, что детерминированная воронка важнее догадок модели.
- LLM post-processing теперь учитывает:
  - `funnel_state.lead_stage=p0_manager_only`;
  - `next_step_type=manager_only_p0`;
  - `known_slots` как источник запрета на повторный запрос класса/предмета/формата.
- Daily metrics дополнены счётчиками:
  - `new_leads`;
  - `qualified_leads`;
  - `next_step_offered`;
  - `manager_handoffs`;
  - `reasked_known_data`.
- Добавлены/расширены unit tests на воронку, сводку менеджеру, prompt, LLM guards, runtime persistence, store idempotency и metrics.

## Что намеренно не делалось

- Не запускался полный v8/v8 dynamic пакет.
- Не запускались ASR, Resolve+Analyze или тяжёлые batch-скрипты.
- Не было live-write в AMO/Tallanto/CRM.
- Не менялась схема SQLite: новые поля пишутся в существующие JSON-поля.
- Не менялся клиентский identity-policy для вопросов “вы кто / как вас зовут / вы бот?”.

## Формальный статус

- `formal_pass`: да.
- `semantic_pass`: `PASS_WITH_NOTES`.
- `pilot_ready`: только для продолжения внутреннего сотруднического пилота после перезапуска ботов и targeted-прогонов.
- `production_ready`: нет.
