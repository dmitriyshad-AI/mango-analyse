# TZ-14 Branch Replant And Polish Report

Date: 2026-06-12

## Scope

Executed the task from `tasks/_inbox_codex/2026-06-12_TZ14_branch_replant_and_polish_PROMPT_for_D4.md`.

Work was done in a clean worktree:

`/Users/dmitrijfabarisov/Projects/Mango-analyse-tz14-replant`

Original dirty worktree was not switched or cleaned.

No AMO/Tallanto/CRM write was executed. No ASR/R+A was launched. `stable_runtime` was not touched.

## Replant

Created branch from current local `main`:

`codex/tz14-d4-crm-amo-tallanto-replant`

Cherry-picked exactly the six TZ-14 commits, preserving order:

| Old commit | New commit | Subject |
| --- | --- | --- |
| `1a8f149f` | `d405cd3d` | TZ14 step1 AMO duplicate snapshot |
| `c7c52bb8` | `13d27ada` | TZ14 step2 AMO new lead scan |
| `1d3ebd12` | `c7515cb0` | TZ14 step3A contact card dry run |
| `a6162d7e` | `91c1ca4b` | TZ14 step4 Tallanto live card |
| `38f795b2` | `35d07b6b` | TZ14 step0 AMO token report |
| `fb77f36e` | `4f3d940a` | TZ14 final report steps 1-4 |

Replant branch tip before merge:

`4f3d940a6dd68111e388f1de2ea93ecbe05c9195`

Forbidden-path diff check for replant branch: no matches for `channels/`, `pilot_context_assembly`, `draft_loop`, `amo_wappi_`, `scripts/run_amo_wappi_draft_loop.py`, or `stable_runtime/`.

## Merge

Merged into `main` with merge commit:

`f5a57281 Merge TZ14 D4 CRM AMO Tallanto`

Merge diff contained only TZ-14 files:

- `scripts/run_tz14_amo_step1_snapshot.py`
- `scripts/run_tz14_amo_step2_scan.py`
- `scripts/run_tz14_amo_step3_contact_cards.py`
- `src/mango_mvp/amocrm_runtime/tallanto_api.py`
- `src/mango_mvp/amocrm_runtime/tallanto_context.py`
- `src/mango_mvp/existing_clients/*`
- `tasks/_done/2026-06-12_TZ14_*`
- `tests/test_existing_clients_amo_step*.py`
- `tests/test_tallanto_api.py`
- `tests/test_tallanto_live_card.py`

Forbidden-path diff check for merge commit: no matches.

Full pytest immediately after merge:

`3049 passed, 5 skipped, 1 warning`

## Polish Commits

- `99778b30` - `TZ14 polish grade text formatting`
- `8a0c5779` - `TZ14 polish WhatsApp brand tags`
- `42aef80f` - `TZ14 polish scan run output folders`
- `92e5ef10` - `TZ14 polish document page limit rationale`

Final `main` head:

`92e5ef10cfd437bc9d9e8d43ee6275282124e66c`

Final diff from pre-merge `main` has no matches for forbidden paths.

## Polish Counters

### 1. Grade Text

Fixed duplicate wording like `класс кл` in Step 2 notes and Step 3 contact cards.

NEG:

- `grade="7 класс"` stays `7 класс`;
- `класс кл` and `класс класс` are blocked by tests;
- numeric grade renders as `8 класс`.

Test:

`tests/test_existing_clients_amo_step2_scan.py tests/test_existing_clients_amo_step3_contact_cards.py` -> `12 passed`

### 2. WhatsApp Brand Tags

Changed WhatsApp bot-context brand tag format from bare `unpk` to `brand:unpk`.

Also changed future WhatsApp import so it writes `brand:<brand>` tags instead of bare brand tags.

Real local retrofit on ignored `tz12_working_batch3/customer_timeline.sqlite`:

- dry-run planned changes: 40,034 WhatsApp `bot_context_chunks`;
- apply changes: 40,034 WhatsApp `bot_context_chunks`;
- timeline events changed: 0;
- Telegram changed: 0;
- repeated apply changes: 0;
- SQLite `PRAGMA quick_check`: `ok`.

Local backup before apply:

`product_data/customer_profiles/tz12_working_batch3/customer_timeline.sqlite.backup_before_brand_tag_20260612`

NEG:

- other channels are not changed;
- Max/non-supported channel is not changed in synthetic test;
- repeated apply is idempotent;
- `whatsapp`, `message`, and `channel_shared:true` tags are preserved.

Test:

`tests/test_retrofit_channel_brand_tags_in_timeline.py tests/test_import_whatsapp_export_to_timeline.py` -> `9 passed`

### 3. Scan Run Folders

Added CLI run-folder allocation for TZ-14 Step 1/2/3 scripts.

Repeated CLI runs with the same `--out-root` now write under separate folders:

`<out-root>/run_YYYYMMDDTHHMMSSZ[_NN]`

NEG:

- second run receives a separate folder;
- first run folder and its `summary.json` are not changed;
- absolute and relative base paths are supported.

Test:

`tests/test_existing_clients_tz14_run_roots.py tests/test_existing_clients_amo_step1_snapshot.py tests/test_existing_clients_amo_step2_scan.py tests/test_existing_clients_amo_step3_contact_cards.py` -> `20 passed`

Compile check:

`py_compile` for the three TZ-14 CLI scripts and `run_roots.py` -> passed.

### 4. Page Limit Rationale

Added report line explaining `page_limit=10`.

Reason: it was an operator safety choice for connector response size, not an AMO API cap.

Confirmation: the completed Step 1 run fetched 1,390 contact pages and 801 lead pages without partial-output rejection.

NEG:

- report grep confirms the line exists in Step 1 and final TZ-14 reports;
- secret grep over changed report files found no secret values.

## Final Tests

Full pytest after all polish commits:

`3053 passed, 5 skipped, 1 warning`

Warning: local `urllib3` reports LibreSSL instead of OpenSSL. This is unrelated to TZ-14 changes.

## Semantic Review

Verdict: `PASS_WITH_NOTES`.

What passed:

- TZ-14 code is now planted on clean `main`, without foreign Wappi/draft_loop commits.
- CRM/Tallanto safety boundary is preserved: read-only only.
- Manager-facing Step 2/3 text no longer creates the obvious `класс кл` artifact.
- WhatsApp brand tags are aligned with Telegram for bot retrieval.
- Repeat scan outputs no longer overwrite the first run folder.

Non-blocking risks:

- Step 2 duplicate draft rows can still appear in a new run folder because the dry-run journal remains run-local. This is acceptable for no-write dry-run artifacts, but live note/task creation remains forbidden.
- The real `tz12_working_batch3` SQLite was modified locally as requested; it is ignored by git and backed up locally.

Required manual gate:

Dmitry must review the 20-family Step 3A package before any Stage B/live CRM writeback decision.

## Git Status

Clean worktree status after final pytest before this report commit:

`## main...origin/main [ahead 447]`

No untracked/generated artifacts in the replant worktree.

Original worktree `/Users/dmitrijfabarisov/Projects/Mango analyse` remains dirty from parallel work and was not cleaned.
