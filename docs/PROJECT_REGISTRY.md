# Project Registry: Mango

Last updated: 2026-06-26

This file is the repository-local project map. It separates four different states that must not be mixed:

- `main`: local repository branch. For the exact current hash, run `git rev-parse --short main`.
- `origin/main`: remote branch, currently older than local main.
- `live`: currently deployed bot runtime. It is not the same thing as `main`.
- `branch`: a candidate worktree/branch, source branch, or audit branch. After merge, keep branches until cleanup is explicitly approved.

Do not treat a branch result, a one-shot live smoke, or a green test run as production readiness. For bot, CRM, AMO, Tallanto, and customer text, green tests mean `formal_pass`; business review is still required before live expansion.

## Current Truth

| Area | Current State | Evidence |
|---|---|---|
| Local `main` | Current local `main` includes `TELEGRAM_INTENT_MODEL_LED`, P0 three-class detector, Wappi watch package, and this registry | `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`; run `git rev-parse --short main` |
| Main worktree folder | `/Users/dmitrijfabarisov/Projects/Mango analyse` is not the clean main worktree; it is on `codex/tz135-direct-wow-tone` and dirty | `git status --short --branch` in main folder |
| Origin | `origin/main=43134ae`; local `main` is ahead of origin. Do not assume remote is current. | `git rev-parse --short origin/main`; `git rev-list --count origin/main..main` |
| Live Telegram bot | Runs from the venue/autonomy live branch, not from local `main` | `Mango_live_4caa5eb_release_venue_autonomy` reports and runtime passport history |
| Live writes | No AMO/Tallanto/CRM/customer write may be run from this registry alone | Project safety boundary |

## Feature Status

| Feature | Status | What It Means | Next Gate |
|---|---|---|---|
| `TELEGRAM_INTENT_MODEL_LED` | In local `main` | Model-led intent is part of `pilot_gold_v1` in local main. It was merged cleanly without venue/autonomy/KB payload. | Needs deployment decision if live should move to this main line. |
| P0 detector: three classes | In local `main` | Handles refund/legal classes and narrows false `refund` on benign “снять стресс/усталость”. Integration tests: targeted `254 passed`; full pytest `3651 passed, 5 skipped`. | Needs business regрейд before live rollout. |
| Wappi watch package | In local `main` | Includes 50-message Wappi context, AMO-events resolver, note format fix, stabilization ops, and default `flex` service tier. Integration tests: targeted `254 passed`; full pytest `3651 passed, 5 skipped`. | Live watch requires a separate live-write gate. |
| Wappi current live process | Not currently alive as main loop | Read-only process check saw no active `run_amo_wappi_draft_loop.py`; only watchdog process was present. Last known context-watch heartbeat: `2026-06-25T17:46:09Z`, `bot_calls=0`, `processed=0`, `auto_resolver_counts.not_enabled=92`. | After merge/regрейд: controlled watch launch with passport, heartbeat, daily report, and quality table. |
| AMO note write allow-all | Server-side setting, not repository code | `CRM_AMO_NOTE_ALLOW_ALL_LEADS=1` must be set in AI Office if broad note writes are intended. Repo code alone cannot remove server `403 lead is not allowlisted`. | Server deployment/config check before live-write expansion. |
| Customer Timeline: Mango calls | Production append was executed by D4; source remains `mango_processed_summary` | Reports show no `source_system='mango_call'`, allowed_for_bot remains false for raw manager chunks. | Wait for remaining pending calls; decide on semantic duplicates; rebuild profiles/summaries if needed. |
| AMO incremental | Branch/test-copy work exists, production apply not done in this plan | Current prod DB sha changed after call appends; old apply plan sha is stale. | Fresh apply plan, backup, process check, and separate “да” before any prod write. |
| AMO cards | Source-label cleanup fixed manager-visible noise | Branch `codex/etap1-crm-card-assembler`; dry-run after cleanup: payload noise 0, 3/5 would write after approval, 2/5 blocked by anti-clobber. | Pick safe rows, fresh readback plan, explicit write permission. |
| Bot-safe memory | Still off for Wappi/live by default | Earlier A/B showed no harm, but low benefit because Wappi-chat to customer coverage and dossier quality were weak. | Improve dossier usefulness and linkage before enabling broadly. |
| Venue/autonomy layer | In live branch, not in local `main` | `FACT_VENUE_SCOPE` and `AUTONOMY_SCOPE_PRECISION` are not part of local `main`. | Separate merge plan; do not accidentally pull via unrelated branches. |

## Active Branches Worth Tracking

| Branch / Worktree | Purpose | Current Gate |
|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_watch_on_main` / `codex/wappi-watch-on-main` | Source branch for Wappi controlled watch package | Merged into local `main`; keep for audit until cleanup decision. |
| `/Users/dmitrijfabarisov/Projects/Mango_p0_three_classes_on_main` / `codex/p0-three-classes-on-main` | Source branch for P0 three-class detector | Merged into local `main`; keep for audit until cleanup decision. |
| `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards` / `codex/etap1-crm-card-assembler` | AMO card assembler and dry-run cleanup | Writeback still gated; no live-write. |
| `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors` / `codex/tz-c-nightly-cursors` | AMO incremental / nightly cursor work | Fresh prod plan required; previous sha stale. |
| `/Users/dmitrijfabarisov/Projects/Mango_project_registry_docs_current` / `codex/project-registry-docs-current` | Source branch for repo-local registry | Merged into local `main`; keep for audit until cleanup decision. |

## Merge Order

Current safe order:

1. Controlled Wappi watch launch after separate live-write gate.
2. AMO card writeback only for explicitly approved rows after fresh dry-run/readback.
3. AMO incremental production apply only after fresh backup/apply plan and separate “да”.
4. Venue/autonomy merge as a separate integration, not as a side effect of P0/Wappi.

## Wappi Launch Gate

Before any persistent Wappi watch:

- Confirm exact branch/commit in runtime passport.
- Confirm `DRAFT_LOOP_AUTO_RESOLVER=1` only after AMO-events resolver is present.
- Confirm `TELEGRAM_BOT_SAFE_CRM_CONTEXT=0` unless a new memory A/B proves value.
- Confirm model/reasoning and `MANGO_CODEX_SERVICE_TIER=flex` unless speed is explicitly needed.
- Confirm AI Office note write policy: allowlist or `CRM_AMO_NOTE_ALLOW_ALL_LEADS=1`.
- Run dry-run against fresh incoming messages.
- Start live-write only with heartbeat, daily report, quality table, kill-switch, and readback.
- Client sends must remain 0.

## AMO Card Write Gate

Before any AMO card write:

- Fresh preview from the current production Customer Timeline.
- Identity gate: AMO contact/lead must match the selected customer.
- Field allowlist: only AI fields.
- Anti-clobber: do not overwrite fields changed since expected-before snapshot.
- Payload quality: no source labels such as `mango_processed_summary`, no raw JSON, no service ids, no brand mix.
- Explicit row list and explicit write approval.
- Readback after write.

## Timeline Apply Gate

Before any production Customer Timeline write:

- Fresh sha of production DB.
- Online SQLite backup before import.
- Process/WAL check.
- Apply to staged copy first.
- Source-system invariant check.
- Idempotency repeat check.
- Rollback path with verified backup.
- Separate explicit approval for the actual production apply.

## Known Non-Goals For This Registry

- It is not a launch command.
- It does not authorize any live write.
- It is not a replacement for `PROJECT_NOW.md`, task reports, audit packs, or deployment runbooks.
- It should be updated after each accepted merge or production decision.
