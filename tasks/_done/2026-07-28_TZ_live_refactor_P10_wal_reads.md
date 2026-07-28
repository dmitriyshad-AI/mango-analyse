> DONE 2026-07-28 20:46 | ветка main | codex

> TAKE 2026-07-28 20:36 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/family_gold.py, src/mango_mvp/customer_timeline/family_graph.py, tests/test_customer_timeline_family_gold.py, tests/test_customer_timeline_family_graph.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_family_gold.py tests/test_customer_timeline_family_graph.py tests/test_customer_timeline_store.py
Семантический-аудит: нет

# P10: семейные чтения должны видеть SQLite WAL

## Цель

Убрать `immutable=1` из двух read-only входов семейного контура и использовать
существующий `customer_timeline_readonly_uri`, чтобы видеть подтверждённые строки
в активном WAL.

## Приёмка

- `family_gold` и read-only `family_graph` видят незачекпоинченную WAL-запись.
- Write-путь family_graph не меняется.
- Целевой и полный pytest зелёные.

## СТОП

- Если исправление требует записи в реальную БД или изменения схемы.
