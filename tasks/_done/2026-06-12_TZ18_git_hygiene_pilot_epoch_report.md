# TZ-18 git hygiene pilot epoch report

Date: 2026-06-12.

## Scope

Executed only git hygiene for the pilot epoch.

Explicitly not touched:
- AI Office server allowlist;
- `DRAFT_LOOP_AUTO_RESOLVER`;
- the postponed batch of 20 auto-pairs.

## Stop window

Dmitry approval for the evening stop window was used.

Stopped processes:
- `mango_draft_loop`;
- `mango_draft_loop_watchdog`;
- `mango_public_pilot_bots_r4`.

Notes:
- `mango_draft_loop` first needed exact screen id because `screen -S mango_draft_loop -X quit` was ambiguous with `mango_draft_loop_watchdog`.
- A stale `.git/index.lock` appeared during the first commit attempt. It was 0 bytes, no live git/editor process existed, and it was removed to continue git operations.

## Commits created before merge

- `5e1ec418` — `Document pilot epoch operating policy`
- `e1942229` — `Ignore pilot runtime artifacts`
- `cf53e6d0` — `Snapshot D1 audit backlog for pilot epoch`
- `b83308a3` — `Snapshot task queue for pilot epoch`
- `9f28dfe4` — `Add pilot support analysis scripts`
- `d7185073` — `Snapshot knowledge base artifacts for pilot epoch`
- `6775ed7d` — `Snapshot financial model working files`

Excluded from D1 commit:
- `D1_audit_backlog/kb_intake_20260610/questions_all_clients_2026-06-10.jsonl`
- `D1_audit_backlog/kb_intake_20260610/questions_clean_2026-06-10.jsonl`

Reason: raw client-question exports contained URL parameters with `secret_key`-like strings. They remain local and are now ignored.

## Merge

Merged `codex/tz14-d4-crm-amo-tallanto` into `main`.

Merge commit:
- `923d9680` — `Merge pilot epoch git hygiene and draft loop work`

Conflict policy:
- TZ14 polish conflicts were resolved in favor of `main` to preserve fresh run-root/output-folder/formatting/page-limit polish.
- Unique draft-loop/Wappi changes from the branch were preserved.

Deleted branch:
- `codex/tz14-d4-crm-amo-tallanto`

Not touched:
- `Mango-analyse-tz16` worktree / `codex/tz16-d4-profiles-v7-rerun-tail`;
- all other unique feature branches;
- stash entries.

## Secret issue and mitigation

During post-commit review, `tallanto_postman_collection(1).json` was found to contain a non-empty `tallanto_token`.

Mitigation:
- file removed from git tracking with `git rm --cached`;
- local file was not deleted;
- `.gitignore` now ignores `tallanto_postman_collection*.json`;
- protective commit created:
  - `6c9da838` — `Stop tracking local Tallanto Postman export`

Important: the token remains in local unpushed history in commits:
- `9f28dfe4`
- `d7185073`
- `6775ed7d`
- `923d9680`

Therefore `git push origin main` was not executed. A separate history-cleaning decision is required before pushing `main`.

## Tests

Full pytest on final `main`:

```text
3071 passed, 2 skipped, 1 warning in 39.70s
```

Warning:
- `urllib3` / LibreSSL warning from local Python environment.

## Restart and checks

Restarted:
- `mango_draft_loop`;
- `mango_draft_loop_watchdog`;
- `mango_public_pilot_bots_r4`.

Draft loop heartbeat after restart:
- `status=ok`;
- `auth_error_count=0`;
- `stop_active=false`;
- `processed=0`;
- `retried_pending=0`;
- `bot_calls=0`;
- `auto_resolver_counts.not_enabled=120`.

Public bots heartbeat after restart:
- `status=polling`;
- brands: `foton`, `unpk`;
- model: `gpt-5.5`;
- reasoning: `high`;
- snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.

Operational note:
- the ignored runtime launcher still pointed to r4 immediately after restart;
- it was updated in `.codex_local/.../public_pilot_bots_r4_launcher.sh` to r4.1 and public bots were restarted again;
- this is runtime-only, not a git commit.

`getme`:
- `ok=true`;
- Foton bot: `foton_intellegence_bot`;
- UNPK bot: `mipt_AI_bot`.

`scripts/check_public_bot_live.py`:
- Foton: `ok=true`, no fallback, facts retrieved.
- UNPK: bot was live and answered without fallback, but check returned `ok=false` because the checker hard-codes Foton online prices `29 750 / 47 250` for the `physics_online` check; UNPK r4.1 correctly answered with `37 000 / 59 000`. Classified as `measurement_bug` in the live-check script, not a restart failure.

Retro-draft check after restart:

```json
{
  "draft_events_since": 0,
  "retro_violations": 0,
  "pairs_with_not_before": 2
}
```

Watchdog:
- process is running;
- status file currently reports `no_drafts_3_working_hours`;
- this is the business alert "no drafts in 3 working hours", not a process/auth failure.

## Push status

`git push origin main` was skipped.

Reason:
- confirmed secret-bearing `tallanto_postman_collection(1).json` in local unpushed history.

## Cleanup candidates for Dmitry decision

No files/folders were deleted as cleanup.

Candidates only:
- `runs/` — local run artifacts, now ignored;
- `_bundles/` — local bundle artifacts, now ignored;
- `Claude Cowork Space/` — local handoff workspace, now ignored;
- `tallanto_postman_collection(1).json` — local Postman export with token, now ignored and untracked;
- `D1_audit_backlog/kb_intake_20260610/questions_all_clients_2026-06-10.jsonl` — raw client-question export with URL `secret_key`-like strings, ignored;
- `D1_audit_backlog/kb_intake_20260610/questions_clean_2026-06-10.jsonl` — raw client-question export with URL `secret_key`-like strings, ignored.

## Final state

Current branch:
- `main`

Current head before this report commit:
- `6c9da838`

Tracked working tree:
- clean except this report and `.gitignore` change before committing the report.

