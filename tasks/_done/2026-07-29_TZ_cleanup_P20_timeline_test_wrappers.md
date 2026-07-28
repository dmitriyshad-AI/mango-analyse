> DONE 2026-07-29 00:20 | ветка main | codex

> TAKE 2026-07-29 00:15 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/wappi_history_import.py, src/mango_mvp/customer_timeline/full_memory_ingest.py, src/mango_mvp/customer_timeline/bot_safe_summary.py, tests/test_wappi_history_checkpoint.py, tests/test_customer_timeline_bot_safe_summary.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_checkpoint.py tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_full_memory_ingest.py
Семантический-аудит: нет

# P20: удалить три устаревшие Timeline-обёртки

## Цель

Удалить старый счётчик Wappi, неиспользуемый описательный safety-словарь и
производственную тестовую подсказку chunk ID. Сохранить проверки через живые
владельцы `wappi_timeline_state` и `stable_chunk_id`.

## Доказательства

- Уборщик и архитектор подтвердили три DELETE_NOW.
- В производственном коде нет вызывающих.
- Новые владельцы строже либо являются каноническими реализациями.

## Приёмка

- Три определения удалены.
- Тесты не удаляют бизнес-инварианты, а проверяют живые функции напрямую.
- Точечные и полный pytest зелёные.

## СТОП

- Найден живой потребитель или потеря проверяемого инварианта.
- Красный тест.

## Результат

- Удалены 3 устаревшие обёртки и 48 строк нетестового кода.
- Проверки сохранены через строгие текущие владельцы.
- Точечные тесты: 73 passed; полный pytest: 5029 passed, 2 skipped.
- Добавлено строк нетестового кода: 0; удалено: 48.
- Новых файлов: 0; флагов: 0; зависимостей: 0.
- Удаление тестов отвергнуто: полезные инварианты перенесены на живые функции.
