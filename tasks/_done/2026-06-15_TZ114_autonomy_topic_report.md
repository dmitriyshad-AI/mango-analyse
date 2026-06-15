# ТЗ-114 — фикс topic_id для матрицы автономности в direct-path

Дата: 2026-06-15
Ветка: `codex/tz114-autonomy-topic`

## Что изменено

1. В direct-path при `TELEGRAM_DEAL_ACTION_DECISION=1` перед вызовом
   `apply_autonomy_matrix_guard` подставляется тонкий `conversation_intent_plan.topic_id`.
   Старый грубый `result.topic_id` сохраняется в metadata:
   - `direct_path_autonomy_topic_from`
   - `direct_path_autonomy_topic`
   - `direct_path.autonomy_topic_from`
   - `direct_path.autonomy_topic`

2. При `TELEGRAM_DEAL_ACTION_DECISION=0` direct-path не меняется:
   topic не подменяется, `action_decision` не создаётся.

3. `requires_manager_approval` больше не ставится безусловно:
   - `answer_only` и обычный `unknown` не требуют manager approval;
   - P0, `manager_only`/`draft_for_manager` и опасные действия
     (`send_payment_link`, `send_crm_data`, `send_document`, `advance_stage`, `handoff_manager`)
     требуют manager approval.

## Проверки

Точечные NEG:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_deal_action_decision.py \
  tests/test_subscription_llm_draft_provider.py::test_direct_path_deal_action_off_keeps_service_topic_parity \
  tests/test_subscription_llm_draft_provider.py::test_direct_path_deal_action_autonomy_uses_intent_topic

11 passed
```

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3273 passed, 2 skipped, 1 warning in 50.20s
```

## Риски и границы

- `AUTONOMY_MATRIX_SAFE_TOPIC_IDS` не расширялся.
- P0, бренд-гейт и output-gate не тронуты.
- В `main` ветка не влита.
- Смысловой риск низкий: клиентский текст не менялся, меняется только routing/metadata при включённом `TELEGRAM_DEAL_ACTION_DECISION`.
