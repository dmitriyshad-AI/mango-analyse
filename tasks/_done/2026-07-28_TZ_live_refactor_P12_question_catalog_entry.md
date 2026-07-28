> DONE 2026-07-28 23:15 | ветка main | codex

> TAKE 2026-07-28 23:05 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/question_catalog/__init__.py, src/mango_mvp/question_catalog/classifier.py, src/mango_mvp/question_catalog/parameters_registry.py, tests/test_question_catalog_normalization.py, tests/test_question_catalog_contracts.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_question_catalog_normalization.py tests/test_question_catalog_contracts.py tests/test_question_catalog_classifier_v2.py tests/test_parameters_registry_v2.py
Семантический-аудит: да

# P12: прозрачный вход каталога вопросов

## Цель

Убрать жадные переэкспорты из `question_catalog/__init__.py`, разорвать
цикл `__init__ -> classifier -> parameters_registry -> __init__` и сделать владельца
каждого имени видимым прямо в импорте.

## Доказанная граница

- Публичные имена корня пакета используют только два теста.
- Живой код и скрипты импортируют конкретные модули.
- Динамических обращений к публичным именам не найдено.
- Внешний неучтённый Python-потребитель вне репозитория — осознанный низкий риск.

## Приёмка

- Два теста переведены на прямые импорты из модулей-владельцев.
- Внутренние импорты `normalization` не идут через корень пакета.
- Корень пакета не импортирует дочерние модули.
- Цикл исчез; свежий импорт `normalization` не тянет `builder` и `classifier`.
- Целевые и полные тесы зелёны; импорт Wappi-входа зелёный.
- Нет новых флагов, зависимостей, файлов кода и смысловых regex.

## СТОП

- Найден живой потребитель публичного переэкспорта внутри репозитория.
- Изменилось поведение каталога вопросов или клиентские тексты.
- Красный полный тест или Wappi import-smoke.

## Результат

- Удалены жадные переэкспорты; внутренние потребители импортируют модули-владельцы.
- Цикл разорван; импорт `normalization`: 4 модуля / 1625 строк вместо 26 / 16575.
- Точечные тесты: 25 passed; Wappi import-smoke: PASS.
- Полный `pytest`: 5040 passed, 2 skipped, 2 известных предупреждения.
- Итоговый diff: −70 строк; новых файлов, флагов и зависимостей нет.
