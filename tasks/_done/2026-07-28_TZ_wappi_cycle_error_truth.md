> DONE 2026-07-28 16:30 | ветка main | codex

> TAKE 2026-07-28 16:25 | ветка main | codex

Ветка: main
Зоны: scripts/run_amo_wappi_draft_loop.py, tests/test_run_amo_wappi_draft_loop.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_run_amo_wappi_draft_loop.py
Семантический-аудит: нет

# ТЗ: честный heartbeat при падении Wappi draft-loop

## Образ результата

Если один цикл Wappi draft-loop падает с неожиданной ошибкой, процесс продолжает следующий цикл, а `heartbeat.json` честно фиксирует `cycle_error` и тип ошибки. Текст исключения не писать: в нём могут оказаться ПДн или секреты.

## Минимальное решение

1. В уже существующей ветке `except Exception` вызвать уже существующий атомарный writer heartbeat.
2. Ошибка записи heartbeat не должна убивать цикл.
3. Добавить регрессионный тест, который доказывает запись `cycle_error` и отсутствие сырого текста ошибки.

## Не делать

- не включать live-бота;
- не писать в AMO/Wappi;
- не внедрять из внешнего пакета лимит 20 заметок/час, TTL/удаление очереди, AMO-breaker и outcome-report;
- не добавлять новые флаги, файлы кода или зависимости.

## СТОП

- любая попытка запустить live-бота или внешнюю запись;
- появление сырого текста исключения в heartbeat;
- красный точечный или полный pytest.

## Приёмка

- точечный тест зелёный;
- полный pytest зелёный;
- live остаётся `NO_PROCESS`;
- нет сырого `str(exc)` в heartbeat.
