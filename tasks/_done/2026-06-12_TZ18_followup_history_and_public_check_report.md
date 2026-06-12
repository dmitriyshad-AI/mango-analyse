# TZ18 follow-up: history cleanup and public bot live-check

Date: 2026-06-12

## Scope

Follow-up after TZ18:

1. Remove local Postman export from unpushed Git history without touching the local file.
2. Re-check new history before pushing.
3. Fix public bot live-check so expected online prices are read from the active KB snapshot per brand.

The live contours were not stopped.

## Git history cleanup

- Rewrote local `main` range `origin/main..main` with `git filter-branch`.
- Removed only `tallanto_postman_collection(1).json` from tracked history.
- The local working file remains present and ignored.
- Removed the `filter-branch` backup ref and expired local reflogs, then ran pruning GC.

## Public bot live-check fix

Changed:

- `scripts/check_public_bot_live.py`
- `tests/test_check_public_bot_live.py`

Behavior:

- The checker now reads expected online semester/year prices from the provided KB snapshot.
- Brand-specific values are used for validation:
  - Foton r4.1: 29 750 / 47 250.
  - UNPK r4.1: 37 000 / 59 000.

This fixes the false red result where the UNPK bot was checked against Foton prices.

## Checks

Targeted tests:

```text
tests/test_check_public_bot_live.py: 6 passed
```

Full pytest:

```text
3072 passed, 2 skipped, 1 warning
```

Public bot dry-run checks on r4.1:

```text
Foton: ok=true, failures=[]
UNPK: ok=true, failures=[]
```

History checks before push:

```text
Postman export path in origin/main..HEAD: 0 commits
Sensitive value scan over new patch lines: clean
```

Push:

```text
origin/main updated to main
```

## Notes

- No server allowlist changes.
- No auto-resolver changes.
- No writes to AMO or clients.
