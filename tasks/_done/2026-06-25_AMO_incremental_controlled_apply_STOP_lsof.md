# AMO incremental v1 controlled apply: STOP на preflight

Дата: 2026-06-25
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`
Код: `39ef431`

## Статус

Controlled apply НЕ запускался. Preflight остановлен на первом жёстком гейте: боевую SQLite/WAL/SHM держит другой процесс.

Не выполнялось:

- `wal_checkpoint(TRUNCATE)` на боевой SQLite;
- пересчёт sha после checkpoint;
- SQLite backup;
- apply на копию;
- repeat run;
- atomic swap;
- rollback.

AMO/Tallanto/CRM write: `0`.
Боевая `customer_timeline`: не менялась.
Notes whitelist: не расширялся.
`bot_safe_summary`: не создавался.

## Команда-гейт

```bash
SRC="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
lsof "$SRC" "$SRC-wal" "$SRC-shm"
```

## Красный результат

```text
COMMAND     PID             USER   FD   TYPE DEVICE   SIZE/OFF      NODE NAME
com.apple 78095 dmitrijfabarisov 4404r   REG   1,13 2596515840 111280763 /Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite
com.apple 78095 dmitrijfabarisov 4406r   REG   1,13      32768 111646485 /Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite-shm
com.apple 78095 dmitrijfabarisov 4412r   REG   1,13          0 111280765 /Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite-wal
```

Процесс:

```text
PID 78095
/System/Library/Frameworks/Virtualization.framework/Versions/A/XPCServices/com.apple.Virtualization.VirtualMachine.xpc/Contents/MacOS/com.apple.Virtualization.VirtualMachine
```

Он также держит старую тестовую копию:

```text
/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors/product_data/customer_timeline/amo_incremental_testcopy_mapping_fix_20260624_124130/customer_timeline.sqlite
```

## Дополнительная проверка pgrep

`pgrep` также показывает активный live-бот:

```text
PID 93828
python scripts/run_telegram_public_pilot_bots.py --env-file /dev/null --mode poll --brand all
```

При этом `lsof -p 93828` не показал открытых `customer_timeline.sqlite`, то есть бот не был держателем этой SQLite в момент проверки. Главный blocker — PID `78095`.

## Решение

По принятому плану apply остановлен до любых действий с файлом.

Для продолжения нужно:

1. Закрыть/освободить процесс PID `78095`, который держит боевую SQLite/WAL/SHM.
2. Повторить `lsof "$SRC" "$SRC-wal" "$SRC-shm"` и убедиться, что открытых handle нет.
3. Повторить `pgrep` и принять решение по live-боту. Если live-бот должен быть полностью остановлен на окно swap, остановить его до apply.
4. Только после этого заново начать controlled apply с preflight, checkpoint, sha, SQLite backup, apply на копию, проверок и swap.
