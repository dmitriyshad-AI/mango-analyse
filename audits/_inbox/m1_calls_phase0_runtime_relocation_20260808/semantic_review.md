# Semantic review

## Результат для менеджера

Relocation не меняет содержание звонка и не повышает незавершённый звонок до
готового. Полный business digest обеих SQLite, ready seal, manager output hash
и отрицательные поля проверяются до/после. Незавершённая запись продолжает
обработку с тем же `source_call_id`, без второй строки.

## Дубли между Mac

Runbook разделяет роли: M1 — Process A, основной Mac — Process B. Перед
передачей останавливаются все конфликтующие labels и проверяются оба locks.
Повтор relocation является no-op, а повтор Process B для уже принятого sealed
drop остаётся idle. Это не заменяет live-проверку текущего сервера Mango.

## ASR

Фактический caller уже последователен: primary Whisper завершается до GigaAM
backfill; затем последовательно идут Resolve и Analyze. Одновременный запуск
двух ASR не обнаружен, поэтому production orchestration не менялся.

## Граница разрешения

GO относится только к коду и синтетическим тестам. Реальный Mango API, ASR,
Resolve/Analyze, launchd, cutover, stable_runtime и внешние записи остаются
запрещены без отдельного решения Дмитрия.
