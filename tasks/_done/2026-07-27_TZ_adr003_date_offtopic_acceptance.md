> DONE 2026-07-27 18:27 | ветка main | codex

> TAKE 2026-07-27 17:43 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, src/mango_mvp/channels/subscription_llm_parts/support.py, scripts/run_telegram_dynamic_client_sim.py, scripts/run_adr003_flag_acceptance_pair.sh, scripts/run_adr003_semantic_reading_e3_paired.sh, scripts/report_adr003_semantic_frame_eval.py, tests/test_subscription_llm_draft_provider.py, tests/test_telegram_dynamic_client_sim.py, tests/test_report_adr003_semantic_frame_eval.py, tests/test_adr003_flag_acceptance_pair_runner.py, tests/test_adr003_semantic_reading_e3_runner.py, ARCHITECTURE.md, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py tests/test_report_adr003_semantic_frame_eval.py tests/test_adr003_flag_acceptance_pair_runner.py
Семантический-аудит: да

# ADR-003: дата, внешние вопросы и честная парная приёмка

## Цель

Закрыть найденные по сырью дефекты измерения `intent_model_led`, не включая эксперимент в live:

1. передавать одну дату оценки боту, транскрипту и судье;
2. запретить модели выдумывать внешние факты, например прогноз погоды;
3. автоматически доказывать, что целевой механизм выключен в ноге B и реально применён в ноге ON.

## Границы

- Не включать `TELEGRAM_INTENT_MODEL_LED` в live.
- Не менять P0, бренд, ПДн и числовые полы.
- Не запускать локально полный LLM-экзамен: после локальных CPU-тестов его запускает M1.
- Не добавлять новый флаг, зависимость или отдельный погодный модуль.

## СТОП

- Любая потеря P0, бренд-, ПДн- или числового пола.
- Нога B содержит след `intent_model_led`, либо обе ноги получают разные даты оценки.
- Точечные или полные тесты красные.

## Приёмка

- Живой prompt получает валидную дату `YYYY-MM-DD`; при отсутствии явной даты использует локальную текущую дату.
- Экзамен фиксирует одну дату для обеих ног и показывает её судье.
- Внешний вопрос без подтверждённых фактов получает безопасный отказ вместо придуманного ответа.
- Отчёт требует `intent_model_led=0` в B и хотя бы одно применение без словарного разрешения в ON.
- Точечные тесты и полный pytest зелёные.
