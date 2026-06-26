# D1 Mango Plan Integration

Дата: 2026-06-26

Ветка: `codex/mango-plan-integration-20260626`
Base: `main@b543991`

## Что интегрировано

Интеграционная ветка собрана поверх текущего локального `main`, где уже есть `TELEGRAM_INTENT_MODEL_LED` в `pilot_gold_v1`.

Добавлены три независимых блока:

1. P0 three classes:
   - снятие/отмена записи оплаченной смены -> `refund`;
   - перенос-возврат оплаченной смены -> `refund`;
   - претензия по договору/дате/ФИО/паспорту -> `legal`;
   - сужение ложного `refund` на benign фразах вроде "снять стресс/усталость ребёнку".
2. Wappi watch package:
   - расширенное Wappi context window;
   - формат AMO note: сначала содержательный draft, потом техблок;
   - смысл "место" != "места";
   - auto-resolver через AMO events;
   - stabilization ops: passport, daily-report, quality-table, endpoint-only guard, smoke tests, kill-switch;
   - `MANGO_CODEX_SERVICE_TIER` default = `flex`.
3. Repo-local registry:
   - `docs/PROJECT_REGISTRY.md`.

## Scope Check

Дифф к `main` содержит P0, Wappi, registry и отчёты. Venue/autonomy код и большая KB-дельта не попали. `FACT_VENUE_SCOPE` / `AUTONOMY_SCOPE_PRECISION` встречаются только в отчётах/реестре как явно не входящие в local `main`.

## Проверки

Целевые тесты:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_p0_perifraz.py \
  tests/test_answer_safety_classifier.py \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_wappi_draft_loop_ops.py \
  tests/test_wappi_stabilization_smoke.py \
  tests/test_amo_wappi_phase1.py \
  tests/test_conversation_intent_plan.py \
  tests/test_codex_exec_service_tier.py
```

Результат: `254 passed in 1.62s`

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3651 passed, 5 skipped, 1 warning in 80.81s`

`git diff --check`: clean.

## Safety

- Live Telegram bot не трогался.
- AMO/Tallanto/CRM write: 0.
- Клиентам ничего не отправлялось.
- `stable_runtime` не трогался.
- Wappi live-watch не запускался.
- Customer Timeline production apply не запускался.

## Остаточные риски

Это `formal_pass` интеграции. Для запуска Wappi live-write нужен отдельный runtime gate: passport, fresh dry-run, AI Office note-write policy, heartbeat, kill-switch, quality table, readback.

Для AMO cards и AMO incremental production apply всё ещё нужны отдельные write approvals.

## Вывод

Интеграционная ветка готова к регрейду и последующему локальному переносу в `main` без push, если Дмитрий/Claude подтвердят.
