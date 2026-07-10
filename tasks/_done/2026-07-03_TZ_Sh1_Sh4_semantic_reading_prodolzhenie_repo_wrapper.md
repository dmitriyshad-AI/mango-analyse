> DONE 2026-07-03 13:55 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py, scripts/report_adr003_semantic_frame_eval.py, scripts/run_adr003_semantic_reading_e2_triple.sh, product_data/telegram_dynamic_test_sets/, tests/test_semantic_reading.py, tests/test_adr003_regex_understanding_moratorium.py, tests/test_report_adr003_semantic_frame_eval.py, tests/fixtures/, docs/ADR003_SEMANTIC_READING_DECISIONS.md, tasks/_running/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_semantic_reading.py tests/test_adr003_regex_understanding_moratorium.py tests/test_report_adr003_semantic_frame_eval.py tests/test_direct_path_semantic_frame_shadow.py
Семантический-аудит: да

# Repo wrapper: Ш1/Ш2-prep semantic reading continuation

Оригинальное ТЗ не перемещать из Foton:

`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_Sh1_Sh4_semantic_reading_prodolzhenie_i_prompt_D1.md`

Граница исполнения:

- выполнить безопасный Ш1: guard-тесты, floor-фиксы, чистые reader-функции без подключения к маршруту;
- подготовить report-only поддержку Ш2 для offline-agreement;
- не запускать M1, live, profile, P0-floor/preblock;
- не выполнять Ш3/Ш4 без отдельного решения Дмитрия после регрейда.
