# Draft loop live-write r4.1 report, 2026-06-12

## Status

Blocked on AI Office production deploy.

Mango side is ready:

- default KB snapshot switched to `kb_release_20260612_v6_7_staging_r4_1`;
- commit: `e4888b88 Switch default KB snapshot to r4.1`;
- targeted tests: `28 passed`;
- full pytest: `3029 passed, 2 skipped, 1 warning`;
- draft loop generated a fresh r4.1 draft for the allowlisted pair.

The live note write is blocked because production AI Office does not expose:

```text
POST /api/integrations/amocrm/leads/{lead_id}/notes
```

Current public checks:

- `GET /api/health` -> `200`;
- `GET /api/integrations/amocrm/status` with API key -> `200`, AMO OAuth `active`;
- `POST /api/integrations/amocrm/leads/47854947/notes` without key -> `404`;
- same POST with API key -> `404`.

Expected endpoint exists in AI Office `origin/main`:

```text
814db41 Allow AMO note writes for linked Telegram test lead
ea50bd5 Add allowlisted amoCRM lead note endpoint
```

But `api.fotonai.online` is still running an older API image. SSH from this Mac to `api.fotonai.online:22` timed out, and `gh` is not authenticated locally, so I did not deploy. Previous reports also say `/opt/ai-office` on the server has dirty live changes, so blind `git pull && docker compose up --build` is unsafe.

## Pending draft

State key:

```text
ec2eed50-b55f	290027369	18255
```

Lead:

```text
47854947
```

Current status:

```text
note_pending
```

Current config fingerprint:

```json
{
  "tree_hash": "e4888b88",
  "kb_release_dir": "kb_release_20260612_v6_7_staging_r4_1",
  "gold_pack_version": "real_manager_gold_2026-06-08",
  "schema_version": "draft_loop_config_fingerprint_v1_2026_06_10"
}
```

The loop was not started persistently because every retry would fail with the missing AI Office endpoint.

## Wappi profile map

Read-only Wappi profile check:

| Brand | Channel | profile_id | Name | Account |
| --- | --- | --- | --- | --- |
| Foton | Telegram | `ec2eed50-b55f` | `ФОТОН` | `79255002588` |
| UNPK | Telegram | `18b255b8-7a67` | `УНПК` | `79255076658` |
| Foton | Max | `2952990f-9e4c` | `ФОТОН` | `79255002588` |
| UNPK | Max | `152b441d-81a2` | `УНПК` | `79255076658` |

Current draft-loop allowlist has only one pair:

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

Recent journal contains many `pair_missing` events for other chats, especially profile `18b255b8-7a67` (UNPK Telegram). Those chats are intentionally skipped until the owner provides explicit `(profile_id, chat_id) -> lead_id` pairs.

## Stop and alert

Stop file:

```text
~/.mango_secrets/STOP_DRAFT_LOOP
```

Heartbeat:

```text
~/.mango_local/draft_loop/heartbeat.json
```

Journal:

```text
~/.mango_local/draft_loop/journal.jsonl
```

The heartbeat file exists. Current heartbeat is stale because the persistent loop is not running.

## Next action

Deploy AI Office `origin/main` at `814db41` to production, preserving existing server dirty changes. After that:

1. Recheck note endpoint:
   - without key -> `401`;
   - with key and allowlisted lead `47854947` -> `200`.
2. Run one live retry:
   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
   CODEX_HOME="$HOME/.mango_local/draft_loop/codex_home_fast" \
   python3 scripts/run_amo_wappi_draft_loop.py --once --live-write
   ```
3. Confirm journal transition `note_retried -> note_written` and note id.
4. Only then start persistent screen loop.

