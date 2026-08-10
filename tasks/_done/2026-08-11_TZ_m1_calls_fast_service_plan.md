> DONE 2026-08-11 02:46 | ветка codex/m1-calls-service-integration-20260811 | codex

> TAKE 2026-08-11 | ветка codex/m1-calls-service-integration-20260811 | codex

Ветка: codex/m1-calls-service-integration-20260811
Зоны: src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/productization/capture_staging.py, scripts/, tests/test_mango_calls_m1_bootstrap.py, tests/test_mango_calls_schedule.py, tests/test_mango_calls_two_processes.py, tests/test_productization_capture_staging.py, tests/test_relocate_mango_calls_pipeline.py, docs/, tasks/, requirements-local-whisper.txt, requirements-local-dual-asr.txt, audits/_inbox/m1_calls_phase0_capture_manifest_recovery_20260808/, audits/_inbox/m1_calls_phase0_runtime_relocation_20260808/, audits/_inbox/m1_calls_fast_service_plan_20260811/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_m1_bootstrap.py tests/test_mango_calls_schedule.py tests/test_mango_calls_two_processes.py tests/test_productization_capture_staging.py
Семантический-аудит: да

# Быстрый план постоянной службы звонков на M1

## Цель

Принять в отдельную интеграционную ветку подтверждённо полезную часть работы
M1, зафиксировать минимальную архитектуру службы звонков каждые 30 минут и
передать M1 точное ТЗ на недостающие защитные механизмы. Подготовить понятный
владельцу документ статуса и архитектуры.

## Входит в этот блок

1. Проверка ветки `codex/m1-calls-cutover-20260807` и выбор принимаемых
   изменений без доверия к старым отчётам.
2. Интеграция принятого кода в текущую ветку без изменения live-runtime.
3. Учёт результатов сравнения Whisper/GigaAM в рабочем решении моделей.
4. План единого запуска каждые 30 минут с отдельно проверяемыми стадиями:
   Mango API, ASR, Resolve+Analyze, публикация и готовый пакет Timeline.
5. Подробное ТЗ M1 на закрытие оставшихся классов потери, дублей, зависания,
   неполной публикации и восстановления после сбоя.
6. `.docx` со статусом и архитектурой, визуальная проверка и до трёх раундов
   независимого аудита Claude.
7. Audit pack, безопасные тесты и один коммит текущего блока.

## Не входит

- запуск ASR, Resolve или Analyze;
- установка или запуск службы на M1;
- запись в AMO, Tallanto, CRM, Google или Яндекс Диск;
- перенос аудио и других реальных данных;
- изменение production Customer Timeline;
- удаление старого runtime;
- смена моделей без слепой ручной проверки качества.

## Приёмка

- принято только то, что доказано исходниками и тестами;
- лёгкий capture не ждёт тяжёлый цикл, а один координатор не допускает параллельного ASR;
- повторный цикл без новых звонков почти пуст и не создаёт дублей;
- состояние звонка хранится в базе/манифесте, а не определяется перемещением файла;
- модель Whisper и GigaAM закреплена по точной версии и ревизии;
- публикация не объявляет неполный день полным;
- AMO остаётся выключенной до нескольких дней ручной приёмки РОПом;
- документ понятен владельцу и прошёл смысловой аудит;
- все live и тяжёлые действия перечислены как последующие ручные шаги.

## СТОП

- найдено пересечение с незакоммиченными изменениями другого диалога;
- выбранная часть M1 меняет live-runtime без отдельного разрешения;
- тесты показывают потерю, повторную обработку или публикацию неполного дня;
- документ или ТЗ раскрывает секреты, телефоны, тексты разговоров или иные ПДн;
- для завершения требуется запуск ASR, R+A, службы или внешняя запись.

## Бритва

Переиспользуется существующий Process A/B и его SQLite/manifest. Захват
отделяется от тяжёлого координатора минимально; второй конвейер и четыре
конкурирующие тяжёлые службы не создаются.
