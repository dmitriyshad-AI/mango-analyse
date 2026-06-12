# Draft loop deferred Wappi fetch fix

Date: 2026-06-12

## Problem

Wappi sometimes returns HTTP 400 for `sync/messages/get` with the text:

```text
Команда fetchMessages сохранена для повторной отправки
```

This is a delayed Wappi task, not an auth error and not a bot failure. Before the fix the draft loop treated it as an exception and could stop before writing a fresh heartbeat.

## Fix

Changed `src/mango_mvp/integrations/draft_loop.py`:

- HTTP 400 with `сохранена для повторной отправки` is classified as `deferred_fetch`.
- The current chat is skipped for this cycle.
- The loop continues with other chats.
- The heartbeat remains `status=ok`.
- `summary.deferred_fetch` is written every cycle.
- A `deferred_fetch` journal event is written when this Wappi response occurs.
- Other HTTP 400 responses still raise as before.

No public Telegram bot code was changed.

## Tests

Targeted:

```text
tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py: 36 passed
```

Full:

```text
3074 passed, 2 skipped, 1 warning
```

NEG covered:

- Wappi deferred-400 does not increment auth errors and does not stop the cycle.
- The same chat is picked up on the next cycle.
- A real non-deferred HTTP 400 still raises.

## Live restart

Restarted only the draft loop screen:

```text
mango_draft_loop
```

Did not touch:

```text
mango_public_pilot_bots_r4
mango_draft_loop_watchdog
```

Fresh heartbeat after restart:

```json
{
  "status": "ok",
  "summary": {
    "auth_error": false,
    "auth_error_count": 0,
    "deferred_fetch": 0,
    "processed": 0,
    "bot_calls": 0,
    "stop_active": false
  }
}
```

Live journal note: after restart Wappi did not return this deferred-400 in the observed cycle, so there is no live `deferred_fetch` row yet. The journal event is covered by the new regression test and will appear on the next real deferred Wappi response.
