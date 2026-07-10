> DONE 2026-07-04 16:31 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-04 15:54 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, scripts/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_direct_p0_text_hygiene.py tests/test_dialogue_memory.py tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py tests/test_adr003_flag_acceptance_pair_runner.py tests/test_adr003_semantic_reading_e3_runner.py
Семантический-аудит: да

# Repo-wrapper: ADR-003 profile enablement + Fix1b hardening + Package-2 runner/gate prep

Источник: сообщение Дмитрия от 2026-07-04 с регрейдом Claude #1 трёх M1-пар.

Scope:
- Включить в pilot_gold default-ON только `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` и `TELEGRAM_DIALOG_SUMMARY_ROLLING`.
- `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` оставить default-OFF; только ужесточить corridor.
- Fix1b corridor: не промоутить отрицательные утверждения о существовании; paid context во входе (`чек`/`квитанция`/`скрин оплаты` и близкие подтверждения оплаты) блокирует corridor.
- Обновить pair-runner так, чтобы Package-2 ON-нога могла явно добавить target reading class `intent_actions` поверх профильных classes.
- Обновить inline text health gate / number verification: адресные числа из KB-фактов текущего хода (`30`, `20` и аналогичные) считаются verified только из конкретных источников хода, не из произвольного blob.
- Собрать M1 package/prompt для пары среза-1a, но не запускать live, не пушить, не включать Fix1b, не удалять legacy.

Hard stops:
- Не трогать live runtime, Telegram send, AMO/CRM/Tallanto writes, Wappi.
- Не добавлять `intent_actions` в профильный default.
- Не включать `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` в профиль.
- Не удалять legacy regex/guards.
- Любой P0/brand/fabrication/manager_only lowering при локальных тестах — stop and report.
