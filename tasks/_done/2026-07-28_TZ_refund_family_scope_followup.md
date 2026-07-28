> DONE 2026-07-28 20:15 | ветка main | codex

> TAKE 2026-07-28 20:09 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/manager_dossier.py, tests/test_customer_timeline_manager_dossier.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# ТЗ: закрыть два края семейного возврата

## Цель

После независимого аудита использовать канонический состав семьи из
`family_members_v1` и не считать заменённую запись возврата активным риском.

## Приёмка

- Возврат у уверенно связанного члена семьи блокирует менеджерскую волну, даже
  если у него нет строки ребёнка в `family_links_v1`.
- Заменённое событие возврата не исключает семью из Owner50.
- Старые тестовые схемы без `family_members_v1` не ломаются.
- Полный pytest зелёный.
