> DONE 2026-07-31 04:07 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 03:39 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: scripts/publish_daily_mango_calls_google.py, scripts/run_mango_calls_process.sh, tests/test_publish_daily_mango_calls_google.py, tests/test_mango_calls_schedule.py, pyproject.toml, requirements.txt, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_schedule.py
Семантический-аудит: да

# ТЗ: ежедневный отчёт РОПа как Google-таблица

## Цель

После успешной локальной публикации v3-XLSX на M1 необязательно преобразовывать
его в нативную Google-таблицу в заданной папке Drive, без дублей и потери листов.

## Требования

1. Переиспользовать проверенный XLSX; не собирать строки заново.
2. До сети проверить schema/content/xlsx sha256 из manifest и безопасный путь.
3. Одна пара «день + content hash» создаёт не более одной Google-таблицы; поиск через appProperties.
4. Изменённый завершённый день создаёт новое immutable generation, старое остаётся для отката.
5. Без folder ID и credentials старое поведение Process B не меняется.
6. Credentials читаются только из внешнего файла 0600; путь из Git/Yandex/repo запрещён.
7. В stdout/stderr нет строк отчёта, телефона, email или ФИО.
8. До upload проверить закрытые права папки; после upload обязательно сверить
   mimeType/parents/appProperties, права, значения, типы, формулы и ссылки.
9. До успешной сверки имя карантинное; финальное имя содержит время публикации.

## Приёмка

- dry-run не импортирует google-auth и не делает сеть;
- corrupt manifest/XLSX и path traversal блокируются;
- existing exact generation возвращает reused;
- duplicate exact generation блокируется;
- create использует Google Sheets mimeType, folder parent и appProperties;
- readback mismatch блокируется;
- wrapper вызывает publisher только при полном Google config и успешном exporter;
- целевые тесты и независимый аудит зелёные.

## СТОП

- Не писать в реальный Google Drive/Sheets.
- Не читать и не копировать реальный service-account JSON.
- Не запускать Process A/B, Mango/Tallanto API, ASR или R+A.

## Бритва

Один новый publisher до 150 строк нетестового кода; google-auth — единственная
новая зависимость, потому что ручная JWT-подпись хуже и небезопаснее.
