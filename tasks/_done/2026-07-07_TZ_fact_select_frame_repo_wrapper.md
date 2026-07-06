> DONE 2026-07-07 01:56 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-07 01:22 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, src/mango_mvp/channels/subscription_llm_parts/support.py, src/mango_mvp/channels/subscription_llm_parts/__init__.py, tests/test_subscription_llm_draft_provider.py, tests/test_fact_venue_scope.py, tests/test_semantic_reading.py, tests/test_adr003_regex_understanding_moratorium.py, tests/fixtures/adr003_direct_path_text_patterns_snapshot.json, docs/ADR003_FACT_SELECT_FRAME_DECISIONS.md, audits/_inbox/, tasks/_inbox_codex/, tasks/_running/, tasks/_done/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_fact_venue_scope.py tests/test_semantic_reading.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# Repo-wrapper: выбор фактов по смыслу SemanticFrame

Source-of-truth ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-07_TZ_vybor_faktov_po_smyslu_frame_5_dlya_D1.md`.

Исполнить до стадии готовности к M1-прогону, live не трогать, M1 не запускать.

Короткие рамки:
- добавить trace-only класс `fact_select_read`;
- добавить default-OFF флаг `TELEGRAM_FACT_SELECT_FRAME`;
- использовать SemanticFrame/requested_product для сужения/приоритизации выбора фактов fail-closed;
- закрыть venue-scope дыру keyword fallback;
- сохранить бренд-пол, exact/adjacent, no invented ids, fail-closed;
- найденный полным тестом хвост: нейтрализовать существующий complaint handoff literal без изменения маршрута/правил;
- добавить решения в `docs/ADR003_FACT_SELECT_FRAME_DECISIONS.md`;
- подготовить audit pack.
