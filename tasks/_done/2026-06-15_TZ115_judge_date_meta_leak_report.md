# TZ-115: judge date grounding + meta leak guard

Дата: 2026-06-15
Ветка: `codex/tz115-judge-date-meta-leak`

## Что сделано

1. `src/mango_mvp/channels/humanity_guards.py`
   - Добавлен отдельный `_META_FACT_PHRASE_RE`.
   - `has_meta_leak()` теперь ловит служебные фразы вида «в фактах нет», «нет в данных», «не указано в фактах».
   - `meta_markers_present()` добавляет маркер `fact_phrase_leak`.
   - `_sanitize_humanity_meta_text` не трогался.

2. `scripts/run_telegram_dynamic_client_sim.py`
   - Добавлен `_fact_window_date_keys(fact)`.
   - В индекс судьи добавляется дата из `before_YYYY_MM_DD` в `fact_key`.
   - `valid_until` сам по себе не индексируется; он используется только как сверка с датой из `before_...`.
   - Regex текста бота не менялся.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_humanity_guards.py tests/test_telegram_dynamic_client_sim.py`
  - `103 passed in 1.09s`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py`
  - `464 passed in 7.30s`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3269 passed, 5 skipped, 1 warning in 46.67s`

## Semantic review

`semantic_pass` для узкого safety-фикса:
- клиентские фразы про внутренние «факты/базу/данные» теперь блокируются как служебные;
- обычные фразы «дату уточнит менеджер», «нет занятий по выходным», «нет свободных мест» не блокируются;
- судья перестаёт ошибочно считать выдумкой дату, явно закодированную в ключе факта;
- произвольный `valid_until` без matching `before_...` не легализует выдуманную дату.

## Не делалось

- Не менялось поведение модели.
- Не менялись regex парсинга текста бота.
- Не индексировался сырой `valid_until` всех фактов.
- Не запускались тяжелые batch/ASR/Resolve+Analyze/live-интеграции.
