> DONE 2026-07-21 01:24 | ветка codex/cleanup-sleeping-top10 | codex

> TAKE 2026-07-21 01:01 | ветка codex/cleanup-sleeping-top10 | codex

Ветка: codex/cleanup-sleeping-top10
Зоны: src/mango_mvp/channels/subscription_llm_parts/, src/mango_mvp/customer_timeline/store.py, scripts/m1_watcher.py, scripts/run_adr003_flag_acceptance_pair.sh, tests/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py tests/test_bot_policy_v2.py tests/test_wappi_stabilization_smoke.py tests/test_m1_watcher.py tests/test_adr003_flag_acceptance_pair_runner.py
Семантический-аудит: нет

# Repo-wrapper: чистка спящего кода

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-20_TZ_D3_chistka_top10_spyashego.md`.

Исполнить только доказанно безопасную часть после свежего Graphify и независимого аудита. Не удалять живой `_verified_informational_answer`, semantic-output safety helpers, платёжное ядро Fix1b, manager-action gate и базовый SemanticFrame shadow. Спорные пять SemanticFrame-веток вынести в отдельный вердикт, если они связаны с сохранённым гейтом или живыми инструментами.

## Приёмка

Точечные P0/brand-тесты, импорт-смоук и полный `pytest` зелёные; для каждого удалённого символа нет ссылок в production-коде и операционных скриптах.

## СТОП

Не удалять кандидат, если найден живой вызов, зависимость сохранённого safety-гейта или непонятная динамическая ссылка.
