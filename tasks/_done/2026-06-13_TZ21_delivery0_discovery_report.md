# ТЗ-21 / Поставка 0 — discovery

Дата: 2026-06-13

## Вердикт

Явной точки принятия решения о продающем действии в текущем Telegram dynamic pipeline нет.

Следствие по ТЗ-21: Часть Б заморожена. Эмиттер `action_intent` и action-judge не реализовывались, потому что без отдельного upstream-сигнала они были бы выводом намерения из текста.

## Что есть в коде

- `src/mango_mvp/channels/subscription_llm_parts/contracts.py`: `SubscriptionDraftResult` содержит `route`, `draft_text`, `manager_checklist`, `crm_recommendations`, `manager_followup_required`, `metadata`, но не содержит `action_intent`.
- `scripts/run_telegram_dynamic_client_sim.py`: `run_one_dialog()` пишет в turn `bot_text`, `bot_route`, `bot_topic_id`, safety/checklist/missing facts, `bot_conversation_intent_plan`, `bot_answer_contract`, но не пишет действие сделки.
- `src/mango_mvp/channels/conversation_intent_plan.py`: `next_step_hint` и `selling` описывают намерение/готовность клиента и подсказку ответа, а не действие бота по сделке.
- `src/mango_mvp/channels/new_lead_funnel.py`: `lead_stage` и `next_step_type` дают состояние квалификации и следующий вопрос/проверку; это не команда `send_payment_link`, `book_trial`, `capture_lead` или `advance_stage`.
- `src/mango_mvp/channels/actions.py` и `src/mango_mvp/amocrm_runtime/agent_runtime.py`: есть отдельный слой recommended actions / dry-run action preview, но он не является решением Telegram dynamic bot pipeline и не вызывается как action signal в симуляторе.

## Формат фактических сигналов

- `route`: `draft_for_manager`, `manager_only`, `blocked`, `bot_answer_self`, `bot_answer_self_for_pilot`.
- `conversation_intent_plan`: `primary_intent`, `topic_id`, `answer_policy`, `route_bias`, `next_step_hint`, `selling`.
- `funnel_state`: `lead_stage`, `next_step_type`, `next_best_question`, `missing_slots`.
- post-factum observers: например `tone_sell_prompt`, которые смотрят уже готовый текст.

Ни один из этих сигналов не является закрытым контрактом `action_intent` из ТЗ-21.

## Покрытие по ходам

- `route` возникает на каждом draft-turn.
- `conversation_intent_plan` и `funnel_state` строятся на каждом ходе до генерации текста.
- `tone_sell_prompt` возникает только после текста и поэтому непригоден как источник действия.
- `a2_proactive` покрывает узкий частный слой контакта/созвона, не общий action-intent.

## Канонический словарь AMO stages

Источник: `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_status_catalog.csv`.

Ключ этапа: строго `(pipeline_id, status_id)`, потому что `142/143` повторяются в разных воронках.

Основная воронка для мок-карточек ТЗ-21: `10408062 / Сделки B2C`.

| sort | status_id | stage |
|---:|---:|---|
| 10 | 82257086 | Неразобранное |
| 10 | 82257090 | Принимают решение |
| 20 | 83489762 | Перспектива |
| 30 | 82257094 | Заключение договора |
| 40 | 82257098 | Запись в группу |
| 50 | 82258194 | Ожидание оплаты |
| 60 | 82258198 | Оплата получена |
| 10000 | 142 | Успешно реализовано |
| 11000 | 143 | Закрыто и не реализовано |

Остальные воронки:

- `8938034 / Лиды`: Неразобранное, В работе, Проблема с контактом, Недозвон, Переговоры, Аудит, Успешно, Закрыто и не реализовано.
- `10431046 / Обзвон`: Неразобранное, В работе, Недозвон, Переговоры, Успешно, Закрыто и не реализовано.

## Решение

Для ТЗ-21 реализована только Поставка 1. Часть Б требует отдельного ТЗ на явное решение о действии в пайплайне, отдельное от генерации текста.
