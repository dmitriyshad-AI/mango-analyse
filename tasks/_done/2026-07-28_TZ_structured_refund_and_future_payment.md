> DONE 2026-07-28 20:08 | ветка main | codex

> TAKE 2026-07-28 19:36 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/purchases.py, src/mango_mvp/customer_timeline/derived_signals.py, src/mango_mvp/customer_timeline/manager_dossier.py, tests/test_customer_timeline_stage5_money_ingest.py, tests/test_customer_timeline_derived_signals.py, tests/test_customer_timeline_manager_dossier.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_stage5_money_ingest.py tests/test_customer_timeline_derived_signals.py tests/test_customer_timeline_manager_dossier.py
Семантический-аудит: да

# ТЗ: отличить возврат от списания баланса и запретить будущую оплату

## Цель

Сохранить полезное исправление обычного списания Tallanto, но не допустить в
продажную выборку семью с явным структурным возвратом. Будущая дата оплаты не
должна подтверждать сезонный сигнал.

## Минимальная реализация

1. Один общий предикат точного направления возврата: `refund`, `return`, `возврат`.
2. `school_out`, `out`, `расход` не считать возвратом без отдельного доказательства.
3. Генератор сезонного сигнала и оба менеджерских пути блокируют структурный возврат.
4. Сопоставление оплаты требует дату не позже даты расчёта.

## Запреты

- Не возвращать запрет по агрегату `total_out`.
- Не добавлять смысловые регулярки и не анализировать свободный текст новым правилом.
- Не менять схему БД и не запускать сбор данных.
- Не ослаблять текстовый P0 и opt-out.

## СТОП

Остановиться, если направление возврата нельзя прочитать из структурного поля
события или для исправления требуется миграция рабочей базы.

## Приёмка

- `in + school_out` допускает сезонную покупку.
- `in + refund/return/возврат` не создаёт сезонный сигнал и исключается из Owner50.
- Будущая дата оплаты не подтверждается.
- Обычные P0/opt-out остаются.
- Целевой и полный pytest зелёные; независимый смысловой аудит PASS.
