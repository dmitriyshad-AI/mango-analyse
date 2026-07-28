> DONE 2026-07-28 17:34 | ветка main | codex

> TAKE 2026-07-28 17:30 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/productization/current_runtime.py, tests/test_productization_current_runtime.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_current_runtime.py tests/test_productization_call_processing_readiness.py
Семантический-аудит: нет

# ТЗ: переиспользовать пять готовых помощников готовности звонков

## Цель

`current_runtime.py` уже зависит от `call_processing_readiness.py`, но повторяет
пять его точных функций. Сохранить локальные имена через алиасы и удалить тела:
`_gate`, `_path_from_value`, `_resolve_optional`, `_mapping`/`_dict`, `_int`.

## СТОП

Не объединять `_load_json_if_exists` с `_load_json`: первая функция fail-soft,
вторая поднимает ошибку. Остановиться при различии сигнатуры/исключений, красной
базовой линии или чужих изменениях.

## Приёмка

- Пять имён ссылаются на функции владельца из `call_processing_readiness`.
- Контракт current runtime не меняется.
- Целевой и полный pytest зелёные.
- Новых файлов рабочего кода, флагов и зависимостей нет; код уменьшается.
