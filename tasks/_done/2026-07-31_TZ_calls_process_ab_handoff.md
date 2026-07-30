> DONE 2026-07-31 01:12 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 00:50 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/productization/capture_staging.py, scripts/run_mango_calls_process.sh, tests/test_mango_calls_schedule.py, tests/test_mango_calls_two_processes.py, docs/RUNBOOK.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_schedule.py tests/test_mango_calls_two_processes.py
Семантический-аудит: нет

# ТЗ: передача sealed drop из Process A в Process B

## Цель

Process B должен запускаться по фактической готовности проверенного sealed drop,
даже если Process A завершился partial. Истёкшее ожидание отсутствующей записи
должно переходить в терминальное состояние и не держать каждый следующий цикл
partial навсегда.

## Требования

1. A сообщает downstream_ready только для status ok/partial и drop.status=ready.
2. Shell-wrapper разбирает результат до возврата исходного кода A, запускает B
   ровно один раз при downstream_ready и сохраняет наблюдаемость partial.
3. run_cycle использует тот же контракт.
4. recording retry TTL считается от первого обнаружения created_at.
5. Истёкший retry терминален и идемпотентен; восстановившаяся запись в том же
   цикле не оставляет ложный partial.

## Приёмка

- partial+ready запускает B; partial без ready и failed не запускают.
- Код Process A=1 сохраняется после успешного B.
- Третий неизменный цикл не повторяет terminal transition.
- Тесты зелёные, реальный runtime не запускается.

## СТОП

- Не запускать launchd, Process A/B, ASR или R+A.
- Не менять runtime SQLite/drop.
- Не трогать multiple recording_ref в этом ТЗ.

## Бритва

Фикс до 50 добавленных строк нетестового кода; новых файлов, флагов и зависимостей нет.
