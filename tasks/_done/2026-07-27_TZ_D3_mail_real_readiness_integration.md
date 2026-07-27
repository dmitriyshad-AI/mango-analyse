> DONE 2026-07-27 12:48 | ветка main | codex

> TAKE 2026-07-27 12:33 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/productization/mail_archive.py, scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/run_customer_timeline_mail_download.py, scripts/run_customer_timeline_mail_import.py, tests/test_customer_timeline_mail_pipeline.py, tests/test_customer_timeline_codex_task.py, tests/test_mail_canonical_paths.py, tests/test_mail_real_readiness.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_pipeline.py tests/test_customer_timeline_codex_task.py tests/test_mail_canonical_paths.py tests/test_mail_real_readiness.py
Семантический-аудит: нет

# Почта: единый свежий архив и рабочая идентификация

## Цель

Исправить доказанные причины остановки почтовой цепочки без второго архива или второго импортера:

1. Ночной построитель читает консолидированную базу и все входящие базы одного канонического корня.
2. До чтения проверяются существование, обязательные таблицы, известная схема и отметка манифеста.
3. `mail_link_enrich` получает только реально существующие Tallanto identity-БД; отсутствие всех блокирует сборку явно.
4. Канонический путь объявлен один раз.
5. JSONL с почтовыми темами и выжимками имеет права 0600.

## Не делать

- Не скачивать письма и не обращаться к IMAP.
- Не читать/выводить тела, адреса и темы реальных писем.
- Не писать в live Customer Timeline, AMO, Tallanto или CRM.
- Не считать отсутствие новых писем за 36 часов аварией: живость проверяется по манифесту цепочки в общем staging-прогоне.
- Не создавать второй архив, парсер, курсор или nightly-путь.

## Приёмка

- Дефолтный построитель видит три части канонического архива и дедуплицирует по SHA.
- Неизвестная схема, отсутствующая таблица/штамп или все отсутствующие identity-БД дают fail-loud.
- Оба вызывающих используют один резолвер identity-БД.
- Нет второго литерала канонического корня.
- Выходной JSONL имеет режим 0600.
- Целевой и полный pytest зелёные.

## СТОП

- Любой сетевой вызов или запись в реальную почту/боевую Timeline.
- Попытка вывести ПДн писем в отчёт или тест.
- Красный целевой или полный pytest.
