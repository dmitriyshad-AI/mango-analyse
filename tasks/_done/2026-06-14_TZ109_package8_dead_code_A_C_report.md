# TZ-109 Package 8 A+C report

Дата: 2026-06-14
Ветка: `codex/tz109-package8-dead-code`

## Scope

Сделаны только Часть A и Часть C.

Часть B (`subscription_llm_parts/monolith.py`) запаркована: паритет-инструменты рассинхронены со split, `monolith.py` не трогался.

## Проверки перед удалением

Команда:

```bash
rg -n "tallanto_deal_ranking|tallanto_premature_close|tallanto_matching" src tests scripts || true
```

Результат: внешних импортёров и вызовов не найдено.

## Изменения

- Удалён `src/mango_mvp/amocrm_runtime/tallanto_deal_ranking.py`.
- Удалён `src/mango_mvp/amocrm_runtime/tallanto_matching.py`.
- Удалён `src/mango_mvp/amocrm_runtime/tallanto_premature_close.py`.
- В `AGENTS.md` добавлена заметка: legacy-слои `policy_routing`, `rules_engine`, `answer_quality_rewriter` заморожены для пилота; живой путь идёт через direct path в `subscription_llm_parts/provider.py`.

## Тесты

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3263 passed, 2 skipped, 1 warning in 49.16s`.

## Статус

Части A и C выполнены. Часть B не выполнялась и остаётся запаркованной.
