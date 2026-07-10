> DONE 2026-07-04 15:53 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-04 13:29 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, tests/, docs/, tasks/, product_data/telegram_dynamic_test_sets/, scripts/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_semantic_reading.py tests/test_adr003_semantic_reading_trace.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# Repo-wrapper: Srez-1a Package-2 Intent Output

Primary TZ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-04_TZ_SREZ1_paketa2_intent_plan_plus_funnel.md` v3.

Scope:
- Implement only Srez-1a intent output mechanics.
- Do not implement Srez-1b or `slots_gsf -> known_slots` merge.
- Do not touch live runtime, profile deployment, AMO/CRM/Tallanto writes, or Telegram sending.
- Keep `intent_actions` out of `PILOT_PROFILE_DEFAULT_READING_CLASSES`; enable only by explicit `TELEGRAM_SEMANTIC_READING_CLASSES` in measurement legs.
- Use inline SemanticFrame only at the same provider stage as the legacy guard.
- Keep deletion of legacy computation blocked until green pair + external regread.
