> DONE 2026-07-04 18:21 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-04 18:08 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, src/mango_mvp/channels/answer_quality_rewriter.py, scripts/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_semantic_reading.py tests/test_adr003_semantic_reading_trace.py tests/test_adr003_semantic_reading_e3_runner.py tests/test_subscription_llm_draft_provider.py tests/test_answer_quality_rewriter.py
Семантический-аудит: да

# Repo wrapper: ADR003 Package-2 slices 2-4

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-04_TZ_SREZY_2_3_4_paketa2_odin_zahod.md`.

Исполняется только строительная фаза:

- добавить классы `route_templates`, `rewrite_quality`, `post_semantics` в существующую механику `TELEGRAM_SEMANTIC_READING_CLASSES`;
- не добавлять новые env-флаги;
- не добавлять классы в профиль-дефолт;
- legacy не удалять;
- P0/бренд/деньги/ПДн floor, tone, sanitizer, verifier, authoritative gate не трогать;
- любые route/text-изменения только за явным классом и fail-closed;
- M1 пакет большого замера собрать после локальных проверок.

Если реализация требует новых `re.*`, новых marker-helper таблиц или правки frozen budgets без явного обоснования — остановиться или оставить точку `kept`.
