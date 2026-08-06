> DONE 2026-08-06 09:20 | ветка codex/question-catalog-retired-cut-20260806 | codex

> TAKE 2026-08-06 08:56 | ветка codex/question-catalog-retired-cut-20260806 | codex

Ветка: codex/question-catalog-retired-cut-20260806
Зоны: src/mango_mvp/question_catalog/, scripts/, tests/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_question_catalog_builder.py tests/test_question_catalog_classifier_v2.py tests/test_question_catalog_contracts.py tests/test_question_catalog_extractors.py tests/test_question_catalog_normalization.py tests/test_question_catalog_safety.py tests/test_draft_prompt_builder.py tests/test_run_p0_model_led_m1_eval.py
Семантический-аудит: да

# Question Catalog: удалить старый автономный контур калибровки

## Проблема

Graphify-карта построена на ревизии `85facd09`; отсутствие внешних входов
дополнительно проверено в сыром текущем коде. Найдены 16 файлов прежнего
офлайн-контура Codex/ROP-калибровки. Все входящие связи замкнуты внутри этого
набора; live-бот, runbook, службы, `pyproject.toml` и активные ТЗ его не вызывают.
Результат ROP уже материализован в `themes_taxonomy.yaml`, а понимание смысла и
современные экзамены реализованы другим живым контуром.

## Образ результата

Удалено ровно 16 файлов и 3 623 строки без замещающего кода, флага или заглушки.
Живой классификатор, extractors/contracts/normalization/safety, LLM theme
assigner, taxonomy/parameters, draft prompt и современные P0/ADR/replay-экзамены
остаются. Бот отвечает и проверяется так же, как до удаления.

## Удалить

- `src/mango_mvp/question_catalog/calibration_metrics.py`
- `src/mango_mvp/question_catalog/codex_full_run.py`
- `src/mango_mvp/question_catalog/rebuild_from_predictions.py`
- `src/mango_mvp/question_catalog/rop_policy_import.py`
- `src/mango_mvp/question_catalog/rop_questionnaire.py`
- `scripts/apply_rop_questionnaire_to_catalog_v2.py`
- `scripts/build_question_catalog_stratified_calibration_v2.py`
- `scripts/build_rop_bot_policy_questionnaire.py`
- `scripts/rebuild_question_catalog_from_llm_predictions_v2.py`
- `scripts/run_question_catalog_codex_ab_v2.py`
- `scripts/run_question_catalog_codex_full_v2.py`
- `scripts/run_question_catalog_llm_calibration_v2.py`
- `tests/test_question_catalog_calibration_v2.py`
- `tests/test_question_catalog_codex_full_run.py`
- `tests/test_question_catalog_rop_policy_import.py`
- `tests/test_question_catalog_rop_questionnaire.py`

## Приёмка

1. До удаления доказаны отсутствие внешних импортов и entrypoint/runtime-вызовов.
2. После удаления `rg` не находит импортов пяти удалённых модулей.
3. Сохранённый Question Catalog/draft/P0 набор зелёный.
4. Полная коллекция уменьшается ровно на 17 тестов: 4 984 -> 4 967.
5. Дифф: 0 добавлений рабочего кода, 3 623 удаления; новых файлов, флагов и
   зависимостей нет.
6. Ломатель подтверждает мутацией импорта, что живой тестовый набор не маскирует
   отсутствие сохранённого обязательного модуля.

## Границы

- Данные, taxonomy/parameters, runtime, внешние системы и сообщения не менять.
- Удаление обратимо через Git; архивную заглушку не создавать.
- Если найден хотя бы один внешний живой вход, соответствующий файл не удалять.

## СТОП

- Graphify и сырой код расходятся по внешнему импортёру или entrypoint.
- После удаления падает сохранённый живой Question Catalog/draft/P0 набор.
- Коллекция уменьшается не на ожидаемые 17 тестов.
