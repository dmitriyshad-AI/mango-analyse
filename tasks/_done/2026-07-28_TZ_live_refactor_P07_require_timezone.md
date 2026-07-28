> DONE 2026-07-28 22:19 | ветка main | codex

> TAKE 2026-07-28 22:13 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/telegram_pilot_store.py, tests/test_exact_runtime_dedup_contract.py, tests/test_single_owner_registry.py, tests/test_telegram_pilot_store.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_exact_runtime_dedup_contract.py tests/test_single_owner_registry.py tests/test_telegram_pilot_store.py
Семантический-аудит: нет

# P07: один владелец проверки часового пояса

## Цель

Удалить байт-идентичное тело `telegram_pilot_store.require_timezone` и сохранить
имя импортом `channels.contracts.require_timezone`.

## Границы

- Не трогать `parse_datetime`: он нормализует время, а не требует пояс.
- Не менять старые встроенные проверки и другие слои в этом пакете.

## Приёмка

- Локальное имя является прямым алиасом владельца.
- Naive/aware/None и текст ошибки ведут себя как раньше.
- Защитный предел определений снижен с 2 до 1.
- Целевой и полный pytest зелёные; рабочий diff отрицательный.

## СТОП

- Любое расхождение результата или исключения останавливает пакет.
- Не менять runtime, хранилища и внешние данные.
