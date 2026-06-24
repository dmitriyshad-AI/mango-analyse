# AMO incremental v1: controlled apply plan для боевой customer_timeline

Дата: 2026-06-24
Статус: план подготовлен, apply НЕ запускался.
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`
Код: `ccf4bea` (`Improve AMO event customer mapping`)

## Жёсткая граница

Без отдельного явного «да» от Дмитрия:

- не писать в боевую `customer_timeline`;
- не запускать apply-команду на боевой папке;
- не запускать apply, если боевую SQLite читает или пишет другой процесс;
- не трогать AMO/Tallanto/CRM write;
- не расширять notes whitelist;
- не создавать `bot_safe_summary`.

AMO/Tallanto/CRM write в плане: `0`.

## Что именно будет применяться

Источники только GET:

1. `/api/v4/leads` с `filter[updated_at][from]`
2. `/api/v4/contacts` с `filter[updated_at][from]`
3. `/api/v4/events` с `filter[created_at][from]`

События `/events`:

- импортируются chat/mail events: `incoming_chat_message`, `outgoing_chat_message`, `incoming_mail`, `outgoing_mail`;
- `common_note_added` остаётся явным остаточным gap: без notes whitelist нет тела заметки, а в тестовом срезе почти все note events не имели customer mapping через текущие `updated_at` cards;
- notes endpoint не используется.

Двухфазная логика зафиксирована:

1. fetch/import `leads` + `contacts`;
2. reload `identity_links`;
3. fetch/import `/events`;
4. ambiguous не склеивать;
5. unmatched оставить unmatched.

## Текущее состояние боевой SQLite ДО apply

Файл:
`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`

Read-only sha ДО:

```text
sha256=3cdd2b8e10c5488768baa4c5cec8af29c8397ad24f0f21c04e7353538771921c
size=2.4G
mtime=2026-06-21 12:10
```

Текущие counts ДО:

```text
amocrm_snapshot|amo_contact_snapshot|10997
amocrm_snapshot|amo_deal_stage|5280
mail_archive|email_message|4168
mail_archive_stage2|email_message|30093
mango_processed_summary|mango_call|71962
master_contacts_snapshot|amo_contact_snapshot|16901
tallanto_snapshot|tallanto_student_snapshot|16901
telegram_history|telegram_dialog|147
telegram_history|telegram_message|1230
```

AMO identity links ДО:

```text
amo_contact_id|11353
amo_lead_id|5739
```

Важно: в боевой SQLite сейчас нет таблицы `ingestion_cursors`. Первый controlled apply должен создать её и записать три курсора:

- `amo_leads_updated_at`
- `amo_contacts_updated_at`
- `amo_events_created_at`

## Проверка отсутствия процессов перед apply/swap

Перед созданием backup, перед apply и перед финальным swap убедиться, что боевую SQLite не читает и не пишет ни один процесс. Особенно важно проверить:

- сборщик выжимок;
- card-preview / preview service;
- живой бот;
- любые одноразовые скрипты, открывающие `customer_timeline.sqlite`;
- WAL/shm файлы рядом с SQLite.

Команды ниже не запускались:

```bash
SRC="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"

lsof "$SRC" "$SRC-wal" "$SRC-shm"
pgrep -af "customer_timeline|bot_safe_summary|card-preview|card_preview|run_telegram_public_pilot_bots|run_customer_timeline"
```

Acceptance:

- `lsof` не показывает процессов по `customer_timeline.sqlite`, `customer_timeline.sqlite-wal`, `customer_timeline.sqlite-shm`;
- `pgrep` не показывает активный сборщик выжимок, preview service, live-бота или импортный скрипт, который держит эту БД;
- если что-то найдено — STOP, apply не начинать.

## Backup plan

До apply сделать SQLite `.backup` и manifest. Простое `cp` основной SQLite недостаточно безопасно при WAL, поэтому plan-of-record: `.backup -> apply на копию -> проверки -> atomic swap`. Команды ниже не запускались.

```bash
SRC="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
BACKUP_DIR="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/backups/amo_incremental_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

KNOWN_SHA_BEFORE="3cdd2b8e10c5488768baa4c5cec8af29c8397ad24f0f21c04e7353538771921c"
SHA_BEFORE_NOW="$(shasum -a 256 "$SRC" | awk '{print $1}')"
if [ "$SHA_BEFORE_NOW" != "$KNOWN_SHA_BEFORE" ]; then
  echo "STOP: source DB sha changed before apply: $SHA_BEFORE_NOW != $KNOWN_SHA_BEFORE"
  exit 1
fi

sqlite3 "$SRC" ".backup '$BACKUP_DIR/customer_timeline.sqlite'"
shasum -a 256 "$SRC" "$BACKUP_DIR/customer_timeline.sqlite" > "$BACKUP_DIR/SHA256SUMS.txt"
sqlite3 "$BACKUP_DIR/customer_timeline.sqlite" "PRAGMA integrity_check;" > "$BACKUP_DIR/integrity_check.txt"
```

Manifest ДО должен содержать:

```json
{
  "source_path": "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite",
  "sha256_before": "3cdd2b8e10c5488768baa4c5cec8af29c8397ad24f0f21c04e7353538771921c",
  "sha256_before_recomputed_immediately_before_apply": "$SHA_BEFORE_NOW",
  "backup_sha256": "value from SHA256SUMS.txt",
  "size_before": "2.4G",
  "git_commit": "ccf4bea",
  "apply_allowed_by_dmitry": true,
  "amo_write": 0,
  "tallanto_write": 0,
  "crm_write": 0
}
```

Apply не начинать, если:

- `sha256_before`, пересчитанный непосредственно перед apply, отличается от зафиксированного sha выше;
- `PRAGMA integrity_check` не вернул `ok`;
- backup sha не записан в manifest;
- рабочее дерево не на `ccf4bea` или новее принятого коммита;
- есть непонятная грязь в кодовой зоне;
- `lsof`/`pgrep` показывает активных читателей или писателей боевой SQLite;
- нет отдельного «да» от Дмитрия.

## Apply command на копию, только после отдельного «да»

Apply нельзя делать in-place в боевой файл. Единственная защита runner'а — куда указывает `out-root`, поэтому `out-root` должен быть отдельной apply-копией, а не боевой папкой.

Важно: `--summary-only` **не dry-run**. Этот флаг только сокращает stdout. При `--use-existing-copy` runner пишет в `customer_timeline.sqlite` внутри указанного `--out-root`. У скрипта сейчас нет dry-run режима вообще.

Подготовить apply-копию из backup:

```bash
APPLY_DIR="$BACKUP_DIR/apply_copy"
mkdir -p "$APPLY_DIR"
cp -p "$BACKUP_DIR/customer_timeline.sqlite" "$APPLY_DIR/customer_timeline.sqlite"
```

Запустить incremental apply только на эту копию:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_customer_timeline_amo_incremental.py \
  --out-root "$APPLY_DIR" \
  --use-existing-copy \
  --since "2026-06-20T00:00:00+00:00" \
  --page-limit 50 \
  --max-pages 10 \
  --sleep-sec 0.4 \
  --summary-only
```

Параметры повторяют принятый тестовый прогон. Если нужен другой lower bound или больше страниц, это отдельное решение до запуска.

После успешных проверок на apply-копии сделать финальную проверку отсутствия процессов и только потом атомарно заменить боевой файл:

```bash
# STOP, если lsof/pgrep что-то показывает
lsof "$SRC" "$SRC-wal" "$SRC-shm"
pgrep -af "customer_timeline|bot_safe_summary|card-preview|card_preview|run_telegram_public_pilot_bots|run_customer_timeline"

STAGED="$SRC.apply_ready"
cp -p "$APPLY_DIR/customer_timeline.sqlite" "$STAGED"
shasum -a 256 "$STAGED" > "$BACKUP_DIR/SHA256_APPLY_READY.txt"
sqlite3 "$STAGED" "PRAGMA integrity_check;" > "$BACKUP_DIR/integrity_check_apply_ready.txt"

mv -f "$STAGED" "$SRC"
```

После swap сразу выполнить read-only sanity checks на новом боевом файле: sha, `PRAGMA integrity_check`, counts, raw chunk safety.

## Проверки после apply

Перед apply на копию сохранить baseline `event_id`, чтобы потом отдельно проверить только новые события этого apply:

```bash
sqlite3 "$APPLY_DIR/customer_timeline.sqlite" \
  "SELECT event_id FROM timeline_events WHERE source_system='amocrm_event' ORDER BY event_id;" \
  > "$BACKUP_DIR/amocrm_event_ids_before_apply.txt"
```

### 1. Counts до/после по типам

До apply counts уже зафиксированы выше. После apply на копию выполнить:

```sql
SELECT source_system, event_type, count(*)
FROM timeline_events
GROUP BY source_system, event_type
ORDER BY source_system, event_type;
```

Ожидания:

- появится `amocrm_event|amo_note|N`, где `N > 0`;
- `amocrm_snapshot|amo_contact_snapshot` и `amocrm_snapshot|amo_deal_stage` могут вырасти или обновиться;
- `master_contacts_snapshot`, `tallanto_snapshot`, `telegram_history`, `mail_archive`, `mango_processed_summary` не должны измениться этим runner'ом.

Для accepted test-copy последнего прогона было:

```text
amocrm_event|amo_note|203
amocrm_snapshot|amo_contact_snapshot|11402
amocrm_snapshot|amo_deal_stage|5449
```

В apply-копии фактическое число может отличаться, потому что `/events` живой и окно страниц движется. После atomic swap те же counts повторить на боевом файле.

### 2. Repeat = 0 дублей

Сразу после apply на копию повторить ту же команду второй раз с теми же параметрами и тем же `$APPLY_DIR`.

Acceptance:

- `second_run.changed_customer_count == 0`;
- новые `amocrm_event` не создаются;
- возможны только dedupe/overlap-дубликаты в отчёте;
- counts по `amocrm_event` и raw chunks не растут при повторе.

### 3. Raw chunks закрыты от бота

SQL:

```sql
SELECT chunk_type, allowed_for_bot, requires_manager_review, count(*)
FROM bot_context_chunks
WHERE source_system = 'amocrm_event'
GROUP BY chunk_type, allowed_for_bot, requires_manager_review;
```

Acceptance:

```text
amo_event_raw|0|1|N
```

и `N == count(amocrm_event|amo_note)`.

### 4. Event -> customer привязка по новым event_id и по всему набору

Проверять два слоя:

1. новые `amocrm_event`, созданные именно этим apply;
2. общий набор всех `amocrm_event` после apply.

Рекомендуемый readback-скрипт:

```python
import json, sqlite3
import os
from pathlib import Path

db = str(Path(os.environ["APPLY_DIR"]) / "customer_timeline.sqlite")
baseline_path = Path(os.environ["BACKUP_DIR"]) / "amocrm_event_ids_before_apply.txt"
baseline_event_ids = set(baseline_path.read_text(encoding="utf-8").splitlines())
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

bad_all = []
bad_new = []
rows = con.execute("""
SELECT event_id, customer_id, record_json
FROM timeline_events
WHERE source_system='amocrm_event'
""").fetchall()

for row in rows:
    record = (json.loads(row["record_json"]).get("record") or {})
    entity_type = record.get("entity_type")
    entity_id = str(record.get("entity_id") or "")
    link_type = "amo_lead_id" if entity_type == "lead" else "amo_contact_id" if entity_type == "contact" else None
    if not link_type:
        bad_all.append((row["event_id"], "missing_entity_type"))
        if row["event_id"] not in baseline_event_ids:
            bad_new.append((row["event_id"], "missing_entity_type"))
        continue
    found = con.execute(
        "SELECT count(*) FROM identity_links WHERE customer_id=? AND link_type=? AND link_value=?",
        (row["customer_id"], link_type, entity_id),
    ).fetchone()[0]
    if not found:
        bad_all.append((row["event_id"], entity_type))
        if row["event_id"] not in baseline_event_ids:
            bad_new.append((row["event_id"], entity_type))

new_count = sum(1 for row in rows if row["event_id"] not in baseline_event_ids)
print({
    "checked_all": len(rows),
    "checked_new": new_count,
    "bad_all": len(bad_all),
    "bad_new": len(bad_new),
})
assert not bad_new
assert not bad_all
```

Acceptance:

- `bad_new=0`;
- `bad_all=0`;
- `checked_new > 0`, если `/events` реально дал новые события.

### 5. Body status и common_note gap

Из JSON-отчёта apply проверить:

- `event_body_status.event_only`;
- `event_body_status.note_body_missing`;
- `fetch.amo_events_created_at.common_note_added_mapping_diagnostics`.

Ожидание:

- chat/mail events импортируются как `event_only`;
- `common_note_added` может остаться `normalized=0` или низким;
- если нужны тела заметок, это отдельный notes whitelist, не часть этого apply.

## Rollback plan

Если после apply на копию не проходит repeat, readback или chunk-safety:

1. Не делать atomic swap.
2. Сохранить failed apply-copy для разбора:

```bash
mv "$APPLY_DIR/customer_timeline.sqlite" "$BACKUP_DIR/customer_timeline.failed_after_amo_incremental.sqlite"
```

Если проблема обнаружена уже после swap:

1. Остановить любые потребители timeline.
2. Убедиться через `lsof`, что файл никто не держит.
3. Вернуть backup атомарно:

```bash
ROLLBACK_READY="$SRC.rollback_ready"
cp -p "$BACKUP_DIR/customer_timeline.sqlite" "$ROLLBACK_READY"
sqlite3 "$ROLLBACK_READY" "PRAGMA integrity_check;"
mv -f "$ROLLBACK_READY" "$SRC"
shasum -a 256 "$SRC"
sqlite3 "$SRC" "PRAGMA integrity_check;"
```

Sha после rollback должен совпасть с `sha256_before`.

Source/report файлы `amo_incremental_sources/`, `amo_incremental_report.json`, `amo_incremental_journal.jsonl` оставить как артефакты разбора или переместить в backup-директорию; на runtime они не должны влиять после восстановления SQLite.

## Итоговый критерий готовности после controlled apply

Controlled apply можно считать формально успешным, если:

- backup manifest создан и sha ДО зафиксирован;
- перед apply и swap подтверждено, что никто не читает/пишет боевую SQLite;
- apply выполнен на копии, не in-place;
- atomic swap выполнен только после всех проверок на копии;
- AMO/Tallanto/CRM write = 0;
- `amocrm_event` добавился;
- repeat run = 0 новых изменений;
- все `amocrm_event` имеют валидную привязку к `identity_links`;
- все raw chunks `allowed_for_bot=0`, `requires_manager_review=1`;
- `common_note_added` остаточный gap явно отражён в отчёте;
- rollback путь проверяем и backup не повреждён.
