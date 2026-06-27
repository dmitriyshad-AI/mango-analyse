# TZ142 direct-path closing manual port report

Date: 2026-06-27

## Scope

Manual port of the safe `tz142` closing-fix delta from source commit `8185242` onto current `main` after the earlier `tz137` port commit `74301b9`.

Source relation checked:

- `589009e` is an ancestor of `8185242`.
- `8185242` is exactly one commit above `589009e`.
- `8185242` adds the direct-path closing hook and tests, but also contains profile auto-enable for `TELEGRAM_TONE_CLOSE_DETECT`.

## What was ported

- Direct-path result now passes through `apply_tone_close_detect_layer(...)` after deal-action and direct-keyword-fallback layers.
- `apply_tone_close_detect_layer(...)` now suppresses CTA text for direct-path close results by detecting either `metadata["direct_path"]` or `direct_path_model`.
- Added regression tests for:
  - closing after direct-path product facts;
  - not cutting a real follow-up product question;
  - replacing a cautious direct-path handoff without adding phone/call CTA.

## What was intentionally not ported

- Did not add `TELEGRAM_TONE_CLOSE_DETECT` to `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- Did not change `tone_block.close_detect_enabled(...)` defaults.
- Did not enable flag B or any `tz137`/`tz142` feature in live.

Reason: source audit found flag B still has a P0 regression, and Claude's bundle explicitly requires `TONE_CLOSE_DETECT` to remain OFF until regrade and explicit approval.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile \
  src/mango_mvp/channels/subscription_llm_parts/provider.py \
  src/mango_mvp/channels/subscription_llm_parts/post_layers.py \
  tests/test_subscription_llm_draft_provider.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py \
  -k 'tone_close_detect or direct_path_tone_close'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Results:

- targeted close/direct-path tests: `11 passed, 516 deselected`
- full pytest: `3672 passed, 5 skipped, 1 warning`

## Source audit notes

Independent audit confirmed:

- worktree `Mango_tz142_flagb_closing` was clean at `codex/tz142-flagb-closing-fix @ 8185242`;
- `8185242` is `tz137` plus one closing-fix commit;
- useful delta is the direct-path close hook/helper/tests;
- profile auto-enable must not be ported yet;
- source reports show B ON still had hard failures (`p0_mishandled`, `made_a_promise`).

## Status

Formal pass only. No production verdict.

Next required step: Claude #1 semantic/source regrade before enabling `TELEGRAM_TONE_CLOSE_DETECT` anywhere.
