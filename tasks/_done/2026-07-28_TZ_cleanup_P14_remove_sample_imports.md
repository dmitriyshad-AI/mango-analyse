> DONE 2026-07-28 23:32 | ветка main | codex

> TAKE 2026-07-28 23:26 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/contact_control_sample_import.py, src/mango_mvp/customer_timeline/deal_aware_sample_import.py, scripts/build_customer_timeline_contact_control_sample.py, scripts/build_customer_timeline_deal_aware_sample.py, tests/test_customer_timeline_contact_control_sample_import.py, tests/test_customer_timeline_deal_aware_sample_import.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_store.py
Семантический-аудит: да

# P14: удалить два устаревших пробных импорта Timeline

## Цель

Удалить два майских одноразовых контура построения локальных пробных баз
вместе с их единственными скриптами и тестами.

## Три доказательства

1. Глобальный поиск нашёл только два модуля, их скрипты, их тесты и исторические отчёты.
2. Модули отсутствуют в обязательном ночном процессе, deploy, RUNBOOK и канонических документах.
3. Импорт шести рабочих точек входа не загрузил эти модули и не создал подключений к SQLite/сети.

## Приёмка

- Шесть файлов удалены без `_attic` и копий; откат через Git.
- В живом коде, скриптах, тестах, deploy и канонических документах нет ссылок.
- Импорт шести рабочих точек зелёный; целевые и полные тесы зелёны.
- Схема и рабочая база Timeline не меняются.

## СТОП

- Любая живая или служебная ссылка за пределами удаляемой шестёрки.
- Красный импорт любой рабочей точки или красный полный тест.

## Результат

- Шесть файлов и 2881 строка удалены напрямую; копий и архива нет.
- Импорт шести рабочих точек: PASS; SQLite/сеть при импорте не затронуты.
- Целевые тесты: 133 passed; полный `pytest`: 5031 passed, 2 skipped, 2 известных предупреждения.
- Схема и рабочая база Timeline не менялись.
