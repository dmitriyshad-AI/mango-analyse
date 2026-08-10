> DONE 2026-08-07 03:34 | ветка codex/m1-calls-final-handoff-20260807 | codex

> TAKE 2026-08-07 01:33 | ветка codex/m1-calls-final-handoff-20260807 | codex

Ветка: codex/m1-calls-final-handoff-20260807
Зоны: docs/m1_calls_handoff_20260801/, docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md, docs/worktrees_registry.md, scripts/run_mango_calls_process.sh, scripts/run_mango_calls_cycle.sh, scripts/bootstrap_m1_mango_calls.sh, scripts/install_mango_calls_two_processes_service.py, requirements-local-whisper.txt, requirements-local-dual-asr.txt, tests/test_mango_calls_m1_bootstrap.py, tests/test_mango_calls_schedule.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_m1_bootstrap.py tests/test_mango_calls_schedule.py tests/test_mango_calls_remote_handoff.py tests/test_mango_calls_two_processes.py tests/test_export_daily_mango_calls_resolve.py tests/test_publish_daily_mango_calls_google.py tests/test_productization_call_processing_readiness.py
Семантический-аудит: да

# Финальный пакет переноса конвейера звонков на M1

## Цель

Довести существующий единый путь обработки звонков до самодостаточного пакета,
по которому новый Codex на M1 сможет безопасно установить зависимости,
получить данные и секреты вне Git, выполнить пробный цикл, подготовить службу,
провести явное переключение с основного Mac и пройти семисуточный пилот.

## В границах

- единая главная инструкция для M1;
- подробный поэтапный план переноса и отката;
- проверка и минимальное исправление существующего основного скрипта;
- подробное ТЗ для Codex на M1 с измеримыми критериями приёмки;
- безопасные локальные тесты без сетевых запросов и тяжёлой обработки;
- audit pack без персональных данных и секретов.

## Вне границ

- передача или чтение реальных секретов, аудио, транскриптов и SQLite;
- ASR, Resolve, Analyze и любые тяжёлые пакетные запуски;
- установка или запуск launchd;
- запись в Mango, Tallanto, AMO, Google или Яндекс Диск;
- переключение живого процесса между компьютерами.

## Приёмка

1. В комплекте один канонический порядок действий от чистого M1 до пилота.
2. Каждый шаг разделён на read-only, подготовку, dry-run, live-cutover и rollback.
3. Для M1 указан один основной wrapper; старый `run_mango_calls_cycle.sh`
   fail-closed и не обходит его. Неверный SHA, небезопасный runtime-путь и
   ошибочная конфигурация публикации блокируются уже сейчас; закрытый баланс,
   межхостовый lock и проверка передачи являются явной приёмкой Фазы 0.
4. Секреты, базы, аудио и расшифровки не передаются через Git; рабочая база и
   аудио не передаются через Яндекс Диск.
5. XLSX/TXT публикуются только после подтверждённого ready-снимка.
6. Описан проверяемый механизм, после реализации Фазы 0 не позволяющий двум
   хостам одновременно исполнять Process A; текущий `runtime_pass=false`.
7. Есть команды проверки, отката и семисуточный критерий завершения.
8. Точечные тесты зелёные; отдельный агент-ломатель и Claude проверили комплект.
   Вердикт этого ТЗ - только `handoff_ready`, не runtime/canary/cutover.

## СТОП

- Найдена незакоммиченная работа другого исполнителя в заявленных зонах.
- Для завершения требуется чтение секретов, реальных клиентских данных или
  запуск ASR, Resolve, Analyze, launchd либо сетевой публикации.
- Нельзя доказать единственность Process A или безопасный откат двумя хостами.
- Исправление требует второго параллельного конвейера вместо существующего.

## Результат 2026-08-07

- Единый пакет, wrapper, bootstrap, runbook, M1 prompt и подробное ТЗ собраны.
- Полный безопасный набор: `214 passed`, одно предупреждение локального LibreSSL.
- Claude V5: `PASS_WITH_LIMITATIONS` только для handoff в кодовую Фазу 0;
  `runtime/canary/cutover/production=false`.
- Предмержевая read-only сверка: ни одна из трёх launchd-меток звонков не
  загружена; службы не менялись.
- ASR, Resolve, Analyze, перенос реальных данных, launchd и публикации не
  запускались.
- Добавлено/удалено строк и число файлов фиксируются итоговым Git-коммитом;
  новых feature flags и зависимостей нет, второй конвейер не создавался.
