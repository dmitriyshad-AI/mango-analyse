# TZ-20 autoresolver all incoming channels report

Date: 2026-06-13.
Branch: `codex/tz20-autoresolver`.

## Implemented

- Auto-resolver production path now processes a newly identified incoming message instead of only creating an auto-pair and skipping the current turn.
- `auto_resolver_started_at` is persisted in draft-loop state and used as the `not_before_ts` boundary.
  - Incoming messages at or before that boundary are skipped and marked processed.
  - Messages after that boundary can create a draft when the deal is resolved unambiguously.
- Added file cache `profile_id:chat_id -> lead_id` for auto-resolver results.
  - Cache lives outside the repo by default: `~/.mango_local/draft_loop/auto_resolver_cache.json`.
  - Cache entries have TTL, default 24h.
  - Cache hits re-check the lead before use; closed/deleted/brand-mismatched leads invalidate the cache and do not produce drafts.
- Added channel gating for staged rollout:
  - `DRAFT_LOOP_AUTO_RESOLVER_CHANNELS=telegram,max` by default.
  - Can be narrowed to `telegram` for TG-only rollout.
- Added AMO lookup throttling and one 429 retry:
  - `DRAFT_LOOP_AUTO_RESOLVER_THROTTLE_SEC`, default `0.2`.
  - `DRAFT_LOOP_AUTO_RESOLVER_RETRY_AFTER_SEC`, default `3.0`.
  - This covers AMO/MCP lookup errors, including Tallanto/CRM 429 surfaced through that path.
- Existing safeguards remain:
  - Telegram exact numeric ID only.
  - Max phone only from Wappi fields, never from text and never numeric `chat_id`.
  - one contact, one open lead.
  - brand check via deal organization field.
  - 403/allowlist desync quarantines only one pair.
  - AI Office remains the only write transport.

## Server allowlist status

No server change was made.

Proposed format to agree with Dmitry before touching AI Office:

- Switch AI Office note endpoint from a fixed per-lead list to a draft-loop scoped mode:
  - env example: `CRM_AMO_NOTE_ALLOWED_MODE=resolved_draft_pairs`.
  - allow note writes only when request is authenticated by the existing AI Office API key.
  - allow only the existing draft-note endpoint and body shape.
  - keep an emergency server-side kill switch, for example `CRM_AMO_NOTE_WRITES_DISABLED=1`.
  - keep current fixed allowlist as optional emergency override/deny-by-default rollback.
- The local draft loop still writes only after deterministic resolver success and still quarantines a single pair on 403.

## Tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_run_amo_wappi_draft_loop.py tests/test_draft_loop.py
42 passed
```

Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3088 passed, 5 skipped, 1 warning in 46.37s
```

## What was not run

- No live Wappi/AMO write run.
- No AI Office server allowlist/config change.
- No public bot restart.

Reason: TZ-20 explicitly requires agreeing the server-side “all recognized pairs” format before changing the server.

