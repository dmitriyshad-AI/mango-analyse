> DONE 2026-07-28 22:05 | ветка main | codex

> TAKE 2026-07-28 22:01 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/telegram_pilot_context_builder.py, tests/test_exact_runtime_dedup_contract.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_telegram_pilot_context_builder.py
Семантический-аудит: нет

# P05: один владелец одинаковой проверки истинности

## Цель

Заменить байт-идентичную `_truthy` в Telegram builder алиасом на
`pilot_context.truthy`, сохранив локальное имя и поведение.

## Почему это минимум

Telegram builder уже импортирует `pilot_context`; новый модуль не нужен.
Остальные похожие функции имеют другой словарь или трёхзначный результат.

## Приёмка

- `_truthy is pilot_context.truthy`.
- Граничные входы дают прежние результаты.
- Целевой и полный pytest зелёные; рабочий diff отрицательный.

## СТОП

- Не трогать `boolish`, настройки, P0 и трёхзначные парсеры.
- Любое расхождение результата останавливает пакет.
