# Draft loop live-write r4.1 started, 2026-06-12

## Итог

Черновиковый контур Wappi -> bot -> AMO note запущен в боевом режиме для текущего allowlist.

## Mango

Default KB snapshot переключён на r4.1:

```text
e4888b88 Switch default KB snapshot to r4.1
```

Проверки:

```text
targeted: 28 passed
full pytest: 3029 passed, 2 skipped, 1 warning
```

## AI Office

Проблема была в продовом AI Office: AMO OAuth был жив, но endpoint заметок отсутствовал.

До фикса:

```text
POST /api/integrations/amocrm/leads/47854947/notes -> 404
```

Сервер найден по прямому IP:

```text
151.242.88.24
```

Домен `api.fotonai.online` идёт через Cloudflare, поэтому SSH на домен таймаутится.

На сервере `/opt/ai-office` рабочее дерево уже было dirty из-за live MCP/read-only изменений. Обычный `git pull` не выполнялся. Применена точечная серверная дельта note endpoint поверх текущего дерева, без checkout/reset/pull.

Бэкап перед правкой:

```text
/opt/ai-office/backups/amo_note_endpoint_20260612_122331
```

Перезапуск:

```text
docker compose up -d --build api
```

После фикса:

```text
GET /api/openapi.json contains /api/integrations/amocrm/leads/{lead_id}/notes
POST notes without key -> 401
POST notes to non-allowlisted lead with key -> 403
```

API container:

```text
ai-office-api-1 Up
```

## Live write result

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
CODEX_HOME="$HOME/.mango_local/draft_loop/codex_home_fast" \
python3 scripts/run_amo_wappi_draft_loop.py --once --live-write
```

Summary:

```json
{
  "auth_error": false,
  "bot_calls": 1,
  "processed": 1,
  "retried_pending": 1,
  "skipped": 523,
  "stop_active": false
}
```

Pending draft `message_id=18255` was retried and written.

AMO note id:

```text
467813727
```

The same pass also processed a new allowed message `message_id=18254`.

AMO note id:

```text
467813789
```

Pending notes after pass:

```text
0
```

## Running loop

Persistent screen:

```text
mango_draft_loop_r4_1
```

Process:

```text
python3 scripts/run_amo_wappi_draft_loop.py --loop --live-write
```

Heartbeat:

```text
~/.mango_local/draft_loop/heartbeat.json
```

Current heartbeat:

```json
{
  "status": "ok",
  "last_cycle_at": "2026-06-12T09:35:06.864957+00:00",
  "summary": {
    "auth_error": false,
    "processed": 0,
    "retried_pending": 0,
    "skipped": 523,
    "stop_active": false
  }
}
```

Journal:

```text
~/.mango_local/draft_loop/journal.jsonl
```

Poller log:

```text
~/.mango_local/draft_loop/poller.log
```

## Stop procedure

Soft stop:

```bash
touch ~/.mango_secrets/STOP_DRAFT_LOOP
```

Hard stop:

```bash
screen -S mango_draft_loop_r4_1 -X quit
```

Stop file currently:

```text
absent
```

## No-drafts watchdog

A local watchdog was started because the existing draft loop had heartbeat/auth-error stop, but no separate continuous "0 drafts for 3 working hours" alert process.

Persistent screen:

```text
mango_draft_loop_watchdog
```

Status:

```text
~/.mango_local/draft_loop/watchdog_status.json
```

Current status:

```json
{
  "status": "ok",
  "alert": false,
  "latest_draft_at": "2026-06-12T09:29:29+00:00",
  "threshold_seconds": 10800,
  "working_hours_moscow": "9:00-21:00 Mon-Fri"
}
```

## Wappi profile map

Read-only Wappi profile check:

| Brand | Channel | profile_id | Name | Account |
| --- | --- | --- | --- | --- |
| Foton | Telegram | `ec2eed50-b55f` | `ФОТОН` | `79255002588` |
| UNPK | Telegram | `18b255b8-7a67` | `УНПК` | `79255076658` |
| Foton | Max | `2952990f-9e4c` | `ФОТОН` | `79255002588` |
| UNPK | Max | `152b441d-81a2` | `УНПК` | `79255076658` |

Current write allowlist:

```json
[
  {
    "profile_id": "ec2eed50-b55f",
    "chat_id": "290027369",
    "lead_id": "47854947",
    "expected_brand": "foton"
  }
]
```

All other chats are skipped as `pair_missing`.

## 48h candidate inventory

Read-only inventory was written outside the repository:

```text
/Users/dmitrijfabarisov/.mango_local/draft_loop_inventory/wappi_amo_candidates_48h_20260612_093336.json
/Users/dmitrijfabarisov/.mango_local/draft_loop_inventory/wappi_amo_candidates_48h_20260612_093336.csv
```

Summary:

```json
{
  "total_candidates": 36,
  "linked_count": 1,
  "manual_required_count": 35,
  "by_profile": {
    "18b255b8-7a67": {
      "brand": "unpk",
      "count": 27,
      "linked": 0,
      "manual_required": 27
    },
    "ec2eed50-b55f": {
      "brand": "foton",
      "count": 9,
      "linked": 1,
      "manual_required": 8
    }
  }
}
```

No mass AMO notes were written for this 48h inventory. The next safe step is owner approval of explicit `(profile_id, chat_id) -> lead_id` rows.

