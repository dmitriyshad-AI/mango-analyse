> DONE 2026-07-29 00:14 | ветка main | codex

> TAKE 2026-07-29 00:10 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/mail_stage2_enrich.py, src/mango_mvp/customer_timeline/__init__.py, tests/test_customer_timeline_mail_stage2_enrich.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_stage2_ingest.py tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_mail_stage2_visibility.py tests/test_marathon2_mail_summary_enrich.py
Семантический-аудит: нет

# P19: удалить устаревший mail_stage2_enrich

## Цель

Удалить одноразовый обработчик старых писем, его отдельный тест и три
реэкспорта из package facade. Данные писем не менять.

## Доказательства

- Живых вызовов модуля в `src/`, `scripts/`, `deploy/` и канонических документах нет.
- Текущий путь загрузки текста находится в `mail_stage2_ingest.py`.
- Ночные новые письма собирает `build_customer_timeline_nightly_dv2_sources.py`.
- Текущую нормализацию/привязку делает `nightly_incremental.py` и A2v3-путь.
- Уборщик и независимый архитектор дали DELETE_NOW.

## Приёмка

- Удалены модуль, его тест и только три соответствующих реэкспорта.
- Поиск по репозиторию даёт ноль ссылок на удалённый модуль и публичные имена.
- Точечные почтовые тесты и полный pytest зелёные.
- Ни одна SQLite-база и ни один runtime-файл не изменены.

## СТОП

- Обнаружен живой вызывающий или незаменённая обязанность.
- Красный тест.

## Результат

- Удалены устаревший модуль (365 строк), его тест (129 строк) и 8 строк фасада: всего 502 строки.
- Действующие ingest, nightly, visibility и A2v3-пути сохранены.
- Точечные тесты: 51 passed; полный pytest: 5029 passed, 2 skipped.
- Добавлено строк нетестового кода: 0; удалено: 373.
- Новых файлов: 0; флагов: 0; зависимостей: 0.
- Перенос в `_attic` отвергнут: модуль заменён, история доступна в Git.
