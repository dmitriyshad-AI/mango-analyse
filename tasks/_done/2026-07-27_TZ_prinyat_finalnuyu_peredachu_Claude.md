> DONE 2026-07-27 03:58 | ветка main | codex

> TAKE 2026-07-27 01:34 | ветка main | codex

Ветка: main
Зоны: scripts/audit_owner_gate_semantic_sample.py, scripts/build_kb_release_v3_from_claude_handoff.py, product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json, src/mango_mvp/channels/answer_safety_classifier.py, src/mango_mvp/channels/conversation_intent_plan.py, src/mango_mvp/channels/dialogue_memory.py, src/mango_mvp/channels/fact_scope_spec.py, src/mango_mvp/channels/p0_recall_spec.py, src/mango_mvp/channels/semantic_roles.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, src/mango_mvp/channels/subscription_llm_parts/post_layers.py, src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py, src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_answer_safety_classifier.py tests/test_fact_scope_conflict_consolidation.py tests/test_kb_release_v3_compact_contract.py tests/test_adr003_regex_understanding_moratorium.py tests/test_customer_timeline_manager_dossier.py tests/test_audit_owner_gate_semantic_sample.py
Семантический-аудит: да

# Принять финальную передачу Claude без опасных регрессий

## Цель

Принять полезные незакоммиченные изменения Claude в единственную ветку `main`,
предварительно исправив доказанные по сырью регрессии и не смешивая runtime или
внешние данные с кодом.

## Обязательные результаты

1. Явная смена темы «это не про возврат ... вопрос про расписание» не является P0.
2. Реальное требование возврата в той же фразе всегда остаётся P0 и уходит менеджеру.
3. Попытка свести «место» против «места» новым словарём отклонена и удалена:
   она нарушает ADR-003 и не заменяет понимание через SemanticFrame/LLM.
4. Мёртвые новые функции и неиспользуемые импорты не остаются.
5. Продолжение генератора Excel 30 семей проходит отдельные тесты и не выдаётся за
   принятую бизнес-витрину без реального файла и ручной проверки владельца.
6. ADR-003 snapshot обновлён штатным генератором и мораторий остаётся зелёным.
7. Факт чужого продукта не возвращается через fallback; структурный тип факта учитывается.
8. Канонический KB snapshot не содержит дубли фактов/фрагментов/источников и служебную очередь, которые уже лежат рядом в реестрах.
9. Глобальный этап 6 не объявляется завершённым: принят только безопасный фильтр фактов по продукту; остальная миграция остаётся отдельной работой.

## Проверки

- точечный pytest из шапки;
- смежные P0/direct-path тесты;
- полный pytest;
- отдельный смысловой аудит опасных фраз;
- audit pack с сырым выводом и остаточными рисками.

## Приёмка

- все тесты из шапки зелёные;
- все перечисленные P0-примеры дают ожидаемый маршрут через боевые функции;
- вопрос о городском лагере не получает факт ЛВШ и наоборот, включая legacy fallback;
- полный pytest не показывает новых падений;
- независимый аудитор не находит P0/бренд/ПДн/анти-выдумка регрессий.

## СТОП

- любой настоящий возврат проходит мимо `p0_pre_gate`;
- новая логика повышает риск автономного ответа на P0;
- тесты требуют live-write, внешнюю систему или изменение рабочей базы;
- обнаружена новая грязь вне зон ТЗ.

## Не делать

- не менять live, базы и внешние системы;
- не удалять rollback-worktree;
- не создавать новую ветку, флаг или зависимость;
- не объявлять Timeline/Owner30 готовыми по одним тестам.
