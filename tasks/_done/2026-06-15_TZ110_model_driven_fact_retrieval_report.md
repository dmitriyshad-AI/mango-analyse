# TZ-110 model-driven fact retrieval report

Дата: 2026-06-15

## Итог

Реализован экспериментальный срез выбора фактов моделью в direct-path retriever под двумя default-OFF флагами:

- `TELEGRAM_RETRIEVER_NEED_SHADOW`
- `TELEGRAM_RETRIEVER_MODEL_DRIVEN`

Флаги не добавлены в `pilot_gold_v1`.

## Что сделано

- A/shadow: тот же вызов ретривера может вернуть `needed_facts` для диагностики.
- B/model-driven: `required_fact_keys` не передается в prompt ретривера, чтобы модель выбирала факты по смыслу.
- Сквозной `required_fact_keys` для scope guard, autonomy, missing facts и памяти не менялся.
- Сохранены id validation, keyword fallback, scope demotion exact -> adjacent.
- В `dynamic_summary.json` добавлен `fact_retrieval_trace` по каждому ходу.
- Добавлены regression-тесты на default OFF, prompt contract, fallback, scope demotion, summary trace.

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py
564 passed in 8.53s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3270 passed, 2 skipped, 1 warning in 49.06s
```

`git diff --check` чистый.

## Остаточный риск

A-only меняет prompt ретривера, потому что нужно получить `needed_facts`. Без второго LLM-вызова нельзя формально доказать, что модель всегда сохранит те же `exact_ids/adjacent_ids`, что и в старом id-only prompt.

Снижение риска:

- флаг default OFF;
- prompt явно требует выбирать id как в обычном режиме;
- ночной замер должен проверить OFF vs A-only vs B по реальным сценариям;
- риск отдельно описан в audit pack.

## Audit pack

`audits/_inbox/tz110_model_driven_fact_retrieval_20260615/`
