Ветка: codex/next-step-proactivity-port
Зоны: src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/post_layers.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, tests/test_customer_timeline_bot_safe_summary.py, tests/test_bot_safe_runtime_context.py, tests/test_bot_safe_memory_step_guard.py, tests/test_subscription_llm_draft_provider.py, tests/test_bot_safe_direct_path_context.py, docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, docs/worktrees_registry.md, tasks/_running/2026-07-10_TZ_next_step_proaktivnost_phase1_port.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_bot_safe_summary.py tests/test_bot_safe_runtime_context.py tests/test_bot_safe_memory_step_guard.py tests/test_subscription_llm_draft_provider.py tests/test_bot_safe_direct_path_context.py
Семантический-аудит: да

# Phase 1: proactivity next-step safety port

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-08_TZ_Codex_proaktivnost_FINAL_posle_audita.md`.

Фаза 0 уже выполнена в `Foton/2026-07-10_PHASE0_proaktivnost_next_step_sverka_D1.md`.

Выполнить только безопасный Phase 1 slice:

1. Точечно портировать next-step семью из `codex/email-pipeline-restore` / `edf58b68`, не вливая EPR целиком.
2. Убрать прайминг пустого next_step в `bot_safe_summary.py`: не рендерить `Следующий шаг: активный шаг не найден`.
3. Добавить bot-path strip неподтверждённого next_step на чтении, не трогая manager-facing читалки.
4. Добавить мягко-рамочный next-step output guard для фраз вида `Следующий шаг — уточнить класс...` без active next_step.
5. Встроить guard в canonical direct-path chain без потери semantic-frame/read-trace цепочки.
6. Перенести/адаптировать регрессионные тесты из EPR; не запускать live/AMO/Wappi/M1.

Не делать в этом заходе:

- whole merge EPR;
- Фаза 2 / включение памяти live;
- P2 slot/recap provenance gate, кроме если он нужен как минимальная регрессия для next-step;
- форматный трек A1;
- новые профильные default-ON флаги.
