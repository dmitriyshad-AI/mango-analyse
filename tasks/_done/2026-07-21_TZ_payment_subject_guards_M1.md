> DONE 2026-07-21 13:56 | ветка codex/payment-subject-guards | codex

> TAKE 2026-07-21 13:40 | ветка codex/payment-subject-guards | codex

Ветка: codex/payment-subject-guards
Зоны: src/mango_mvp/channels/subscription_llm_parts/, tests/, docs/worktrees_registry.md, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_bot_policy_v2.py tests/test_subscription_llm_draft_provider.py tests/test_wappi_stabilization_smoke.py
Семантический-аудит: да

# Две защиты ответа и приватный M1-экзамен

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-20_TZ_vlitie_chistki_i_2_zashity_k_dengam.md`.

## Реализация

- Переиспользовать существующие `apply_payment_confirmation_guard` и `apply_unstated_subject_guard`.
- Подключить их в живой `SubscriptionLlmDraftProvider.build_draft()` после поздних слоёв и до `apply_authoritative_output_gate`.
- Один новый флаг `TELEGRAM_PAYMENT_SUBJECT_GUARDS`, default OFF; не включать в `pilot_gold_v1` до M1 PASS и отдельного решения владельца.
- Добавить сквозные тесты OFF/ON через `build_draft`, не дублировать ядро guard-ов.
- Собрать полный M1-экзамен на приватной ветке `yandex`; датасеты допускаются только после `pii_scan=0`, в публичный `origin` их не отправлять.

## Приёмка

- OFF сохраняет текущее поведение.
- ON: неподтверждённая/конфликтная оплата уходит в `manager_only`; названный ботом без основания предмет убирается и требует проверки менеджера.
- Безопасные ответы не меняются.
- M1: Wappi 200, risk31 с тремя rejudge, memory OFF/ON, микронабор защит; P0 по большинству 2/3; бренд, подмена, ПДн и рискованные числа равны нулю.

## СТОП

- Не включать флаг в live/profile и не расширять Wappi до M1 PASS и отдельного GO владельца.
- Не отправлять клиентам и не писать в AMO, Tallanto, CRM или боевые базы.
- Не пушить обезличенные диалоги в публичный `origin`; любой ненулевой PII scan блокирует приватную публикацию набора.
