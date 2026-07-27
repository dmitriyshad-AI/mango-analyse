> DONE 2026-07-27 11:34 | ветка main | codex

> TAKE 2026-07-27 11:02 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/customer_timeline/amo_incremental.py, src/mango_mvp/productization/capture_staging.py, scripts/build_mango_call_timeline_increment.py, tests/test_mango_calls_two_processes.py, tests/test_customer_timeline_amo_incremental.py, tests/test_productization_capture_staging.py, tests/test_mango_call_timeline_increment.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_two_processes.py tests/test_customer_timeline_amo_incremental.py
Семантический-аудит: нет

# Интеграция доказанных исправлений AMO и звонков

## Задача

Взять полезные части пакета `Foton/codex_artifacts/amo_calls_real_readiness`,
не принимая его две неполные защиты.

## Требования

- AMO-событие версии карточки получает `event_at=updated_at`.
- Событие звонка сохраняет `audio_path`.
- Process B принимает только sealed drop с полным манифестом:
  `status=ready`, `quick_check=ok`, SHA и размер совпадают с SQLite.
- Отсутствующий, битый или неполный манифест даёт FAIL без импорта и без
  движения курсора.
- Отсутствующая или пустая аудиозапись учитывается и делает Process A
  `partial`, а не ложным `ok`.
- Существующий capture manifest используется как очередь повторного скачивания:
  `downloaded` без реального файла перестаёт считаться завершённым.
- Звонок без записи остаётся в ограниченной очереди повторного опроса, а битый
  непустой файл перекачивается вместо бесконечной повторной проверки.
- Неверная схема пакета и расхождение `rows_selected/events_written` дают FAIL.
- Не запускать ASR, сеть, live и внешние записи.

## Приёмка

- Новые регрессионные тесты зелёные.
- Целевой и полный pytest зелёные.
- Нет новых зависимостей и флагов.

## СТОП

- Остановиться при необходимости менять боевую базу, ASR или внешний API.
