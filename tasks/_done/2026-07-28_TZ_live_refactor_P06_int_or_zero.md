> DONE 2026-07-28 22:12 | ветка main | codex

> TAKE 2026-07-28 22:07 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/customer_context_for_draft.py, tests/test_exact_runtime_dedup_contract.py, tests/test_customer_context_for_draft.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_customer_context_for_draft.py
Семантический-аудит: нет

# P06: один владелец одинакового int-or-zero в channels

## Цель

Заменить байт-идентичный `customer_context_for_draft.int_or_zero` алиасом на
`pilot_context.int_or_zero`, сохранив прежнее имя и исключения.

## Границы

- Не трогать Customer Timeline и CSV-парсеры: другой слой/формат.
- Не исправлять `OverflowError` в этом механическом пакете.

## Приёмка

- Локальное имя является прямым алиасом владельца.
- None/строки/bool/большое число/nan/inf ведут себя как до правки.
- Целевой и полный pytest зелёные; рабочий diff отрицательный.

## СТОП

- Любое расхождение результата или типа исключения останавливает пакет.
- Не менять маршруты, P0, ПДн или внешние данные.
