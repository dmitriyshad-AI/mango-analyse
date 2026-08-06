> DONE 2026-08-06 12:03 | ветка codex/global-code-cut-sales-insights | codex

> TAKE 2026-08-06 11:25 | ветка codex/global-code-cut-sales-insights | codex

Ветка: codex/global-code-cut-sales-insights
Зоны: src/mango_mvp/insights/, src/mango_mvp/question_catalog/builder.py, scripts/build_insight_readiness_report.py, scripts/build_outcome_linkage_report.py, scripts/build_pilot_sales_moments.py, scripts/build_rop_validation_pack.py, scripts/build_sales_insight_knowledge_base.py, scripts/run_pilot_sales_moment_llm_review.py, scripts/merge_pilot_sales_moment_llm_reviews.py, tests/test_insight_readiness.py, tests/test_knowledge_base.py, tests/test_outcome_linker.py, tests/test_llm_review.py, tests/test_llm_review_merge.py, tests/test_pilot_extraction.py, tests/test_rop_validation_pack.py, tests/test_phone_normalization_canonical.py, tests/test_sanitizer_context_exclusions.py, tests/test_question_catalog_builder.py, docs/DATA_MODEL.md, docs/SCRIPT_SAFETY_MATRIX.md, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_sanitizer_context_exclusions.py tests/test_phone_normalization_canonical.py tests/test_tone_score.py tests/test_customer_timeline_next_step_resolver.py tests/test_transcript_quality_stage15_export_gate.py
Семантический-аудит: да

# ТЗ: удалить закрытый Sales Insights reporting island

## Проблема

Старый офлайн-конвейер sales insights содержит отдельные readiness, outcome,
pilot extraction, LLM review, knowledge-base и ROP-report реализации. Текущие
Customer Timeline, Question Catalog, KB release и live direct path его не
вызывают. Контур поддерживает собственные смысловые regex, телефонный helper,
CLI и тестовый остров, увеличивая проект примерно на 7,2 тыс. строк без
наблюдаемой бизнес-пользы.

## Образ результата

В проекте остаются только реально используемые владельцы из пакета insights:
`sanitizers.py`, `tone_score.py` и `__init__.py`. Старый отчётный остров и его
CLI удалены. Защитные тесты санитайзера сохранены в существующем тестовом файле.
Живой draft/Timeline/KB-путь, P0, бренд, ПДн и анти-выдумка не меняются.

## Доказательства до правок

1. Свежий Graphify на `eafa729f` и сырой `rg` не находят внешних runtime-import
   семи удаляемых модулей. Их потребители - только семь wrapper-скриптов и
   замкнутые тесты.
2. В `pyproject.toml`, deploy/config и LaunchAgents entrypoint нет.
3. `phone_identity.py` оборачивает канонический `mango_mvp.utils.phone`.
4. Блок тестов от
   `test_sanitize_answer_normalizes_brand_money_terms_and_personal_data` до
   `test_sanitize_answer_stable_text_keeps_existing_client_safe_parity` в
   `tests/test_knowledge_base.py` содержит 11 независимых тестов живого
   `sanitizers.py`; их нельзя терять.
5. `tests/test_phone_normalization_canonical.py` должен сохранить проверки
   канона и живых wrappers, убрав только удаляемый insight-wrapper.

## Реализация

1. Перенести 11 тестов `sanitizers` в
   `tests/test_sanitizer_context_exclusions.py` без изменения смысла.
2. Удалить семь старых модулей, семь wrapper-скриптов и семь замкнутых тестов.
3. Убрать только insight-wrapper из канонического теста телефона.
4. Удалить устаревшие строки из текущих DATA_MODEL и SCRIPT_SAFETY_MATRIX.
5. Зафиксировать замороженный `enriched_reviews.csv` как обязательный retained
   input Question Catalog: отсутствие файла должно завершать сборку ошибкой, а
   не молча удалять 2337 из 9969 вопросов.
6. Не добавлять replacement, feature flag, dependency, facade или новый файл
   кода.

## Приёмка

- удалены не менее 7 100 строк net;
- импорт `sanitizers` и `tone_score` работает;
- все перенесённые защитные тесты проходят;
- целевой набор зелёный;
- collect-only не содержит удалённых тестов и не падает на импортах;
- полный pytest имеет только тот же известный baseline-класс падений;
- отрицательный контроль доказывает, что тест санитайзера краснеет при подмене
  проверяемого узла;
- `rg` не находит текущих entrypoint удалённых скриптов;
- `formal_pass`, `semantic_pass`, `business_pass`, `data_pass`, `runtime_pass`
  выставлены отдельно.

## Стоп-условия

- найден внешний runtime/deploy/manual entrypoint;
- для удаляемого модуля нет действующего современного владельца смысла;
- потерян хотя бы один защитный тест санитайзера;
- изменился живой клиентский prompt, P0, бренд, ПДн или write-путь.

## Бритва

Три варианта: оставить остров; построить фасад; удалить замкнутый остров. Выбран
третий: он единственный уменьшает код и не создаёт второго владельца функции.
Почему это минимум: только удаления, перенос уже существующих тестов и чистка
двух текущих документов.

## Результат

- Удалено 21 файл замкнутого острова; `sanitizers` и `tone_score` сохранены.
- Баланс: 289 добавлено, 7475 удалено, net -7186 строк.
- Перенесены 11 защитных тестов; добавлены 3 NEG-кейса retained call source.
- Targeted: 122 passed. Full: 4902 passed, 3 skipped и прежние 10 baseline
  failures.
- Реальный read-only retained CSV: 2726 строк, 2337 извлечённых вопросов.
- Audit pack:
  `audits/_inbox/global_code_cut_sales_insights_20260806120139/`.
