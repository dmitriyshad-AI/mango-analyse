# TZ-26 action decision layer report

Дата: 2026-06-14
Ветка: `codex/tz26-action-decision`

## Что сделано

- Добавлен флаг `TELEGRAM_DEAL_ACTION_DECISION`, default OFF.
- Носитель решения: `SubscriptionDraftResult.metadata["action_decision"]`.
- Сырое предложение модели: `SubscriptionDraftResult.metadata["action_proposal"]`.
- Direct prompt при включённом флаге просит модель вернуть только предложение действия из закрытого списка.
- Авторитетное решение строится детерминированным слоем после финальных гейтов:
  - P0/`manager_only` всегда понижает действие до `handoff_manager`;
  - `send_payment_link` разрешается только при подтверждённой цене из фактов, однозначном продукте, явной готовности в последней реплике и отсутствии возражения/ухода;
  - `send_crm_data` требует строгой идентификации и совпадения бренда карточки;
  - рассинхрон текст↔действие понижается до `answer_only`, текст не переписывается.
- `dynamic_dialog_transcripts.jsonl` получил поля `bot_action_proposal`, `bot_action_decision`, `bot_action_decision_action`.
- `dynamic_summary.json` получил блок `action_decision`.
- Сводка менеджеру использует `action_decision`; при `unknown` старая эвристика следующего шага не подставляется.

## Границы

- Бот ничего не исполняет live: поле является рекомендацией менеджеру.
- `TELEGRAM_DEAL_ACTION_DECISION` не добавлен в `pilot_gold_v1`.
- Порог уверенности для продающих рекомендаций числом не задан в ТЗ-21 §7.6; произвольное значение не зашито. В metadata оставлена явная отметка `threshold_configured=false`, а безопасность держится жёсткими предусловиями.

## Проверки

- Целевые тесты: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_deal_action_decision.py tests/test_manager_handoff_summary.py`
  - Результат: `11 passed`.
- Регресс упавших OFF-паритет кейсов после правки точки врезки:
  - Результат: `18 passed`.
- Полный pytest:
  - Команда: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - Результат: `3204 passed, 2 skipped, 1 warning`.

## Остаточный риск

- Численный порог уверенности для продающих действий должен быть добавлен после калибровки ТЗ-21 §7.6. До этого `send_payment_link` защищён только детерминированными предусловиями и модель может только понизить решение.
