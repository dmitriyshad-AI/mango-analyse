# AMO incremental v1 controlled apply: refreshed preflight STOP

Дата: 2026-06-26

## Итог

Apply в боевую Customer Timeline НЕ запускался.

Причина: старый apply-plan от 2026-06-24 устарел после последующих боевых
append-доливов звонков и текущий файл SQLite всё ещё открыт системным
процессом macOS Virtualization.

## Текущий production DB

```text
path=/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite
size=2.5G
mtime=2026-06-25 19:06
sha256=ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8
wal_size=0B
shm_size=32K
```

Старый план ожидал:

```text
sha256=3cdd2b8e10c5488768baa4c5cec8af29c8397ad24f0f21c04e7353538771921c
mtime=2026-06-21 12:10
```

Значит старый `KNOWN_SHA_BEFORE` больше невалиден. Запуск по нему должен
остановиться.

## Current counts

```text
amocrm_snapshot|amo_contact_snapshot|10997
amocrm_snapshot|amo_deal_stage|5280
mail_archive|email_message|4168
mail_archive_stage2|email_message|30093
mango_processed_summary|mango_call|72998
master_contacts_snapshot|amo_contact_snapshot|16901
tallanto_snapshot|tallanto_student_snapshot|16901
telegram_history|telegram_dialog|147
telegram_history|telegram_message|1230
```

Важно: `mango_processed_summary|mango_call` уже `72998`, а в старом плане было
`71962`; это ожидаемо после D4 call append, но требует нового baseline.

## lsof

На момент preflight файл открыт:

```text
COMMAND     PID             USER   FD   TYPE DEVICE   SIZE/OFF      NODE NAME
com.apple 71111 dmitrijfabarisov *961r   REG   ... customer_timeline.sqlite
com.apple 71111 dmitrijfabarisov *714r   REG   ... customer_timeline.sqlite-wal
com.apple 71111 dmitrijfabarisov *716r   REG   ... customer_timeline.sqlite-shm
```

Это похоже на macOS Virtualization/VM holder. Чтению и `.backup` это может не
мешать, но старый план требует `lsof`-чистоту перед финальным swap. Поэтому
боевой apply/swap по текущему плану не запускался.

## Что нужно перед реальным apply

1. Обновить controlled apply plan под fresh baseline:
   - `sha256=ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`;
   - текущие counts выше;
   - git HEAD `f0c5177` или новее.
2. Разделить два режима:
   - **backup/apply на копию** можно делать при read-only holder, если `.backup`
     и `integrity_check` проходят;
   - **atomic swap боевого файла** делать только при `lsof`-чистоте или после
     отдельного решения владельца о безопасном окне записи.
3. Получить отдельное свежее «да» Дмитрия на production apply.
4. Перед apply заново пересчитать sha, снять SQLite `.backup`, записать manifest
   и только затем применять AMO incremental к apply-copy.

## Safety

- AMO/Tallanto/CRM write: 0
- Customer Timeline write: 0
- Боевая SQLite не изменялась этим preflight.
- Notes whitelist не расширялся.
- `bot_safe_summary` не создавался.
