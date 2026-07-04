> TAKE 2026-07-04 04:02 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, src/mango_mvp/channels/dialogue_memory.py, src/mango_mvp/integrations/draft_loop.py, scripts/, tests/, docs/, tasks/, audits/_inbox/, product_data/telegram_dynamic_test_sets/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# ADR-003 ночной марафон: repo-local wrapper

Источник ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-04_TZ_NOCHNOY_marafon_D1_shest_blokov.md`.

Работать только по финальной версии внешнего ТЗ с hash `efec962662c383ebed772a664dd1548052023a727f824a7696ba29f63828097a`.

Жёсткие границы:

- живой бот, Wappi, AMO, Tallanto, CRM, M1 и push не трогать;
- каждый блок отдельным коммитом и отдельным мини-отчётом в `audits/_inbox`;
- локальные замеры только направленческие, semantic_pass утром остаётся за Fable/Claude по сырью;
- если код уже существует — инвентаризация вместо переизобретения;
- если блок упёрся в стоп-критерий или две итерации — зафиксировать стоп и перейти дальше.

Блоки исполнять строго 1→6 по внешнему ТЗ:

1. profile-aware дефолты `TELEGRAM_SEMANTIC_FRAME_SHADOW` и CSV `TELEGRAM_SEMANTIC_READING_CLASSES`.
2. PR-A Fix1b через отдельный wrapper `tasks/tz_pr_a_fix1b_wrapper.md`.
3. PR-B slots_reask как инвентаризация `10bb2f6e+` и доделка только отсутствующего через `tasks/tz_pr_b_slots_reask_wrapper.md`.
4. PR-D rolling dialog summary.
5. `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` для болезни #16 и формулировок.
6. Data-only `docs/ADR003_PACKAGE2_INVENTORY.md`.
