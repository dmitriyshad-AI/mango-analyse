> DONE 2026-07-28 23:56 | ветка main | codex

> TAKE 2026-07-28 23:51 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/crm_card_aggregator.py, src/mango_mvp/services/analyze.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py tests/test_analyze.py tests/test_analyze_xa_safe_pack.py tests/test_run_analyze_ab_test.py tests/test_tz19_analyze_tail_import.py
Семантический-аудит: нет

# P17: удалить мёртвые CRM- и analyze-помощники

## Цель

Удалить шесть невызываемых функций из `crm_card_aggregator.py` и метод
`_analysis_llm_prompt` из `services/analyze.py`.

## Доказательства

- Поиск по Python-коду на текущем HEAD находит только определения.
- Одноимённые функции в `scripts/build_deal_aware_preview_pack.py` являются отдельными локальными реализациями и не импортируют CRM-модуль.
- Живые ветви анализа вызывают `_analysis_prompt_context` напрямую.

## Приёмка

- Удалены только семь невызываемых определений.
- Поиск даёт ноль ссылок на удалённые имена в соответствующих владельцах.
- Точечные и полный pytest зелёные.

## СТОП

- Аудитор находит динамический вызов, monkeypatch или внешний контракт.
- Красный тест либо изменение живого prompt-пути.

## Результат

- Удалены 7 мёртвых определений и 84 строки нетестового кода.
- Независимый аудит: 7 DELETE_NOW, 0 BLOCK; скрытых вызовов нет.
- Точечные тесты: 148 passed; полный pytest: 5031 passed, 2 skipped.
- Добавлено строк нетестового кода: 0; удалено: 84.
- Новых файлов: 0; флагов: 0; зависимостей: 0.
- Переписывание живых функций отвергнуто: для задачи достаточно удаления невызываемого кода.
