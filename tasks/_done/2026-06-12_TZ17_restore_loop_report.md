# TZ-17 restore loop and allowlist path report

Date: 2026-06-12.

## Step 1 — live draft loop restored

Dmitry approval from prompt was used for removing stop files and restarting the loop.

Actions:
- Removed `~/.mango_secrets/STOP_DRAFT_LOOP`.
- Removed `~/.mango_local/draft_loop/STOP_DRAFT_LOOP`.
- Restarted `screen` sessions:
  - `mango_draft_loop`
  - `mango_draft_loop_watchdog`

Current heartbeat:
- `status=ok`
- `stop_active=false`
- `auth_error_count=0`
- `pending_notes_count=0`
- `quarantined_pairs=0`

Journal/state confirmation:
- No retroactive drafts were created after restart for messages with `timestamp <= not_before_ts`.
- `retro_violations=0`.
- The old pair remains configured and not quarantined. There were no fresh inbound messages on the old pair during the verification window, so no new note write was expected.
- The two newer manual pairs also had no new post-restart inbound events; old inbound messages did not generate notes.

## Step 2 — auto-resolver review file

A direct full dry-run with live AMO MCP was attempted with `--chat-limit 0`, empty pairs file, dry-run mode, separate local state:

- Journal: `~/.mango_local/draft_loop_inventory/auto_resolver_full_20260612T125600Z/journal.jsonl`
- Stopped manually after active progress for about 16 minutes to avoid an unbounded foreground run.
- Partial live-MCP journal: `rows=213`, `matched=22`, no AMO writes.

Reason: Wappi has 5012 chats across four profiles, and live AMO MCP resolution per candidate is too slow for foreground full coverage.

To still provide architect review coverage without write risk, I built the permanent review file using full Wappi chat listing plus the fresh local AMO TZ14 snapshot:

- Review file: `~/.mango_local/draft_loop_inventory/auto_pairs_for_review_2026-06-12.json`
- Method: Wappi full chat listing + local AMO snapshot `product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_step1_snapshot.sqlite`, read-only.
- Scope: latest inbound text dialogs only; no AMO writes; no auto-pair writes.
- Dialogs scanned: `5012`
- Private dialogs: `4956`
- Non-private skipped: `56`
- Latest inbound text dialogs: `1270`
- Matched candidates: `76`

Reason counts:
- `matched=76`
- `latest_not_inbound_text=3686`
- `username_only=493`
- `closed_lead=243`
- `no_contact=238`
- `no_active_lead=85`
- `max_phone_missing=72`
- `multi_contact=34`
- `multi_active_lead=13`
- `brand_mismatch=12`
- `shared_phone=4`

The file contains PII and stays outside the repository.

## Step 3 — shared family phone stoplist for Max

Created outside git:

- `~/.mango_secrets/shared_phones_stoplist.json`

Source:
- `product_data/customer_profiles/tz14_amo_step1_full_20260612/common_phone_review.csv`

Result:
- Source rows: `113`
- Valid phones in stoplist: `112`
- Filtered invalid value: `+7`

Note: the TZ mentioned `identity_links ~367`, but current local sources do not confirm that number. The freshest TZ14 AMO profile report gives explicit `possible_common_phone_distinct_parents` rows and is safer for Max auto-resolve. I also checked local `identity_links`; counts differed by snapshot and did not match `~367`.

Code now uses the TZ path `~/.mango_secrets/shared_phones_stoplist.json` by default and keeps backward fallback to legacy `shared_phone_stoplist.json`.

## Step 4 — server allowlist instruction

Created:

- `D1_audit_backlog/INSTRUKCIYA_server_allowlist_update.md`

It documents:
- server-side code allowlist `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`;
- env allowlist `CRM_AMO_NOTE_ALLOWED_LEAD_IDS`;
- backup/edit/restart/check flow;
- 401/403/200 probes;
- what not to touch;
- proposal for a restricted non-root deploy user.

No SSH/server connection was used.

## Code changes

- `scripts/run_amo_wappi_draft_loop.py`
  - default Max shared-phone stoplist path changed to plural `shared_phones_stoplist.json` with legacy fallback;
  - matched auto-resolver candidates now include organization values in `lead_snapshot` for architect review;
  - `--chat-limit 0` now means full paging instead of being coerced to 50.
- `src/mango_mvp/integrations/draft_loop.py`
  - added paged dialog iterator for explicit `chat_limit=0`.
- `tests/test_run_amo_wappi_draft_loop.py`, `tests/test_draft_loop.py`
  - added regression tests for stoplist path fallback, organization snapshot, and full paging mode.

## Tests

Targeted:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_run_amo_wappi_draft_loop.py tests/test_draft_loop.py`
- Result: `34 passed`

Full pytest:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
- Result: `3067 passed, 2 skipped, 1 warning in 47.53s`

## Model calls

TZ-17 work did not call the bot model:
- auto-resolver review: `0` model calls;
- live-loop verification window: `bot_calls=0` because no fresh paired inbound messages arrived.
