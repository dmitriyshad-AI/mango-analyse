# Отчёт: звонки, раздельные процессы и последовательный запуск

Ветка: `codex/tzv-calls-schedule-brand`.

## Что сделано

- Process A: Mango API -> download -> один последовательный внешний worker на стадию Whisper, GigaAM, Resolve, Analyze -> консистентный drop.
- Process B: только drop -> `mango_processed_summary` -> тестовая timeline под single-writer lock.
- Расписание теперь цепочное: только A имеет интервал 1800 секунд; отдельная demand-only служба B запускается только после явного `status=ok` у A.
- `failed`, `deferred`, `locked`, ошибочный JSON или ошибка запуска B останавливаются явно.
- Установщик сохраняет старые plist и loaded-state и восстанавливает их при сбое `bootout`/`bootstrap`.
- `brand_evidence` считается детерминированно как `single`/`both`/`none`, без модели.

## Фактическая проверка

- рабочая база: 293 звонка с расшифровкой; Analyze `done=285`, `pending=8`; dead-letter `0`;
- тестовая timeline: 75 580 событий `mango_call`, 75 580 уникальных `dedupe_key`;
- единственный источник событий звонков: `mango_processed_summary`;
- `quick_check=ok`, последний обработанный звонок: `2026-07-12T15:29:21+00:00`;
- новые bot-context chunks: `allowed_for_bot=0`, `requires_manager_review=1`;
- повтор Process B на неизменном drop: `idle/drop_unchanged`, дублей нет;
- последний capture до смены расписания получил один download-failure и корректно завершился `capture_failed`; B по новой цепочке после такого результата не запускается.

## Расписание на хосте

- `com.mango.calls-process-a`: loaded, `RunAtLoad=false`, интервал 1800 секунд;
- `com.mango.calls-process-b`: loaded, `RunAtLoad=false`, собственного интервала нет;
- legacy `com.mango.calls-two-processes` не загружен;
- backup старых plist: `.codex_local/launchd_backups/20260712T205149`;
- ручной kickstart и дополнительный ASR не запускались.

## Проверки

- целевые тесты после цепочки: `69 passed`;
- полный тест окончательного кода: `4677 passed, 5 skipped`;
- plist: `plutil` успешно;
- независимый аудит GPT-5.5 high: code PASS и post-install PASS.

Остаток: 8 звонков остаются в ручной очереди Analyze/Resolve; это не dead-letter и не исправлялось обходом. Production timeline, `stable_runtime`, AMO, Tallanto и CRM не изменялись.
