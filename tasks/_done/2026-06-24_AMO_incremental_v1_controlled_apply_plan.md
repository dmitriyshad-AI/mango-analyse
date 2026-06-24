# AMO incremental v1: controlled apply plan для боевой customer_timeline

Дата: 2026-06-24
Статус: план подготовлен, apply НЕ запускался.
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`
Код: `ccf4bea` (`Improve AMO event customer mapping`)

## Жёсткая граница

Без отдельного явного «да» от Дмитрия:

- не писать в боевую `customer_timeline`;
- не запускать apply-команду на боевой папке;
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

## Backup plan

До apply сделать файловый backup и manifest. Команды ниже не запускались.

```bash
SRC="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
BACKUP_DIR="/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/backups/amo_incremental_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp -p "$SRC" "$BACKUP_DIR/customer_timeline.sqlite"
shasum -a 256 "$SRC" "$BACKUP_DIR/customer_timeline.sqlite" > "$BACKUP_DIR/SHA256SUMS.txt"
sqlite3 "$BACKUP_DIR/customer_timeline.sqlite" "PRAGMA integrity_check;" > "$BACKUP_DIR/integrity_check.txt"
```

Manifest ДО должен содержать:

```json
{
  "source_path": "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite",
  "sha256_before": "3cdd2b8e10c5488768baa4c5cec8af29c8397ad24f0f21c04e7353538771921c",
  "size_before": "2.4G",
  "git_commit": "ccf4bea",
  "apply_allowed_by_dmitry": true,
  "amo_write": 0,
  "tallanto_write": 0,
  "crm_write": 0
}
```

Apply не начинать, если:

- backup sha не совпал с исходным sha;
- `PRAGMA integrity_check` не вернул `ok`;
- рабочее дерево не на `ccf4bea` или новее принятого коммита;
- есть непонятная грязь в кодовой зоне;
- нет отдельного «да» от Дмитрия.

## Apply command, только после отдельного «да»

Боевая папка используется как `out-root`; `--use-existing-copy` означает, что runner будет писать в уже существующий `customer_timeline.sqlite` в этой папке.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_customer_timeline_amo_incremental.py \
  --out-root "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621" \
  --use-existing-copy \
  --since "2026-06-20T00:00:00+00:00" \
  --page-limit 50 \
  --max-pages 10 \
  --sleep-sec 0.4 \
  --summary-only
```

Параметры повторяют принятый тестовый прогон. Если нужен другой lower bound или больше страниц, это отдельное решение до запуска.

## Проверки после apply

### 1. Counts до/после по типам

До apply counts уже зафиксированы выше. После apply выполнить:

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

В боевом apply фактическое число может отличаться, потому что `/events` живой и окно страниц движется.

### 2. Repeat = 0 дублей

Сразу после apply повторить ту же команду второй раз с теми же параметрами.

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

### 4. Event -> customer привязка по всему apply-набору

Проверять не 10 примеров, а все `amocrm_event`, созданные apply. Рекомендуемый readback-скрипт:

```python
import json, sqlite3

db = "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

bad = []
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
        bad.append((row["event_id"], "missing_entity_type"))
        continue
    found = con.execute(
        "SELECT count(*) FROM identity_links WHERE customer_id=? AND link_type=? AND link_value=?",
        (row["customer_id"], link_type, entity_id),
    ).fetchone()[0]
    if not found:
        bad.append((row["event_id"], entity_type))

print({"checked": len(rows), "bad": len(bad)})
assert not bad
```

Acceptance: `bad=0`.

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

Если после apply не проходит repeat, readback или chunk-safety:

1. Не запускать сборщик выжимок поверх новой timeline.
2. Сохранить повреждённый файл для разбора:

```bash
mv "$SRC" "$BACKUP_DIR/customer_timeline.failed_after_amo_incremental.sqlite"
```

3. Вернуть backup:

```bash
cp -p "$BACKUP_DIR/customer_timeline.sqlite" "$SRC"
shasum -a 256 "$SRC"
sqlite3 "$SRC" "PRAGMA integrity_check;"
```

4. Sha после rollback должен совпасть с `sha256_before`.
5. Source/report файлы `amo_incremental_sources/`, `amo_incremental_report.json`, `amo_incremental_journal.jsonl` оставить как артефакты разбора или переместить в backup-директорию; на runtime они не должны влиять после восстановления SQLite.

## Итоговый критерий готовности после controlled apply

Controlled apply можно считать формально успешным, если:

- backup manifest создан и sha ДО зафиксирован;
- AMO/Tallanto/CRM write = 0;
- `amocrm_event` добавился;
- repeat run = 0 новых изменений;
- все `amocrm_event` имеют валидную привязку к `identity_links`;
- все raw chunks `allowed_for_bot=0`, `requires_manager_review=1`;
- `common_note_added` остаточный gap явно отражён в отчёте;
- rollback путь проверяем и backup не повреждён.
