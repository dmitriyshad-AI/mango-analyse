# TZ-16 Step 1 Replant Worktree Cleanup

Date: 2026-06-12

## Scope

Cleaned up the temporary TZ-14 replant worktree as required by TZ-16.

No AMO/Tallanto/CRM writes were executed. No ASR/R+A was launched. `stable_runtime` was not touched.

## Checks

Canonical `main` before cleanup:

`40c0a9fa TZ14 replant and polish report`

New TZ-16 feature branch:

`codex/tz16-d4-profiles-v7-rerun-tail`

Removed worktree:

`/Users/dmitrijfabarisov/Projects/Mango-analyse-tz14-replant`

Remaining worktrees after cleanup:

- `/Users/dmitrijfabarisov/Projects/Mango analyse` on `codex/tz14-d4-crm-amo-tallanto`
- `/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16` on `codex/tz16-d4-profiles-v7-rerun-tail`

## NEG

- Main dirty working folder was not switched, cleaned, or edited.
- Replant worktree was clean before removal.
- `git worktree list` no longer contains `Mango-analyse-tz14-replant`.
- `test -d /Users/dmitrijfabarisov/Projects/Mango-analyse-tz14-replant` returns `removed`.

## Status

`formal_pass`: yes.

`semantic_pass`: not applicable; this is a git hygiene step.
