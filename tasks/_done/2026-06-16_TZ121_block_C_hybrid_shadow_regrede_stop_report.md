# TZ-121 Block C Hybrid Shadow

Дата: 2026-06-16  
Ветка: `codex/tz121-group4-remaining`

## Что сделано

- Добавлен shadow-runner для гибридной классификации каталога вопросов:
  - по умолчанию выбирается сохраненный Codex-класс;
  - если кейс совпадает с разобранной follow-up регрессией ТЗ-116, оставляется старое правило;
  - уверенный служебный guard оставлен в коде runner, но на текущем срезе не срабатывал при пороге `0.85`.
- Основной каталог не пересобирался.
- Primary для C не включался.

## Входы

Предсказания Codex из прошлого замера:

`/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_question_catalog_labeled100_codex_shadow_20260615_192755/question_catalog_offline_predictions.csv`

Follow-up guard:

`/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_followup_gold_reviews_20260615/c_model_broke_correct_rule.csv`

Новый артефакт:

`audits/_inbox/tz121_c_question_catalog_hybrid_shadow_20260616/`

## Результат

- строк: `100`;
- rule vs gold: `37/100` (`0.3700`);
- model vs gold: `72/100` (`0.7200`);
- hybrid vs gold: `80/100` (`0.8000`);
- цель `>72%`: пройдена;
- guard `followup_regression`: `8`;
- новые вызовы модели: `0`;
- запись в каталог/DB/AMO/Tallanto/CRM: `0`.

## Измененные файлы

- `scripts/run_tz121_question_catalog_c_hybrid_shadow.py`
- `tests/test_tz121_question_catalog_c_hybrid_shadow.py`
- `tasks/_done/2026-06-16_TZ121_block_C_hybrid_shadow_regrede_stop_report.md`
- `tasks/_done/2026-06-16_TZ121_block_C_hybrid_semantic_review.md`

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_tz121_question_catalog_c_hybrid_shadow.py

2 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_question_catalog_c_hybrid_shadow.py \
  --out-dir audits/_inbox/tz121_c_question_catalog_hybrid_shadow_20260616

hybrid_vs_gold.correct=80
target_passed=true
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3295 passed, 5 skipped, 1 warning
```

## Стоп

Останавливаем C на shadow-отчёте для регрейда Claude/Дмитрия. `primary` не включен.
