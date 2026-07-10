# ADR-003 live freeze before deploy

Дата freeze: `2026-07-06T16:43:55+0300`

Режим: read-only. Live-процессы не останавливались, Telegram/Wappi/AMO/CRM/Tallanto write не выполнялся, secrets значения не раскрывались.

## Repo candidate

- Worktree: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`
- Branch: `codex/adr003-semanticframe-migration`
- HEAD: `d0357d790ce75005c5d6bc06e65eb57181bd5e03`
- Git status на момент freeze: только текущий running-TZ был untracked до фиксации этого документа.
- Machine: `MacBook-Pro-Dmitrij.local`, Darwin `25.5.0`, arm64.

## Live Telegram process

- PID: `60227`
- Parent PID: `60226`
- Start: `Mon Jun 29 01:21:14 2026`
- Command: `scripts/run_telegram_public_pilot_bots.py --env-file /dev/null --mode poll --brand all`
- CWD по `lsof`: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`
- Screen: `60224.mango_public_pilot_bots_main_eb6fa0b_reliable_20260629_v2` detached.

Важная нота: screen-имя содержит `eb6fa0b`, а текущий worktree уже на `d0357d79`. Процесс запущен 2026-06-29 и держит код, загруженный при старте. Перед swap нельзя считать, что live-процесс уже исполняет текущий HEAD только потому, что cwd указывает на тот же worktree.

## Telegram heartbeat

Файл: `.codex_local/telegram_pilot_bots/runtime/public_pilot_bots_heartbeat.json`

- Status: `polling`
- Heartbeat time: `2026-07-06T13:43:56+00:00`
- PID in heartbeat: `60227`
- Effective profile: `pilot_gold_v1`
- Draft path: `direct_path`
- Brands: 2
- Model: `gpt-5.5`
- Reasoning effort: `high`
- Snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`
- Profile selfcheck: `ok=true`, failures/warnings empty.

Active guards reported by heartbeat:

- `autonomy_scope_precision`
- `fact_venue_scope`
- `intent_model_led`
- `number_gate_scope_aware`
- `output_sanitizer`
- `p0_model_led`
- `pii_relation_stopwords`
- `presale_pii_memory`
- `presale_safety`
- `prose_model_led`
- `semantic_output_verifier`
- `verifier_handoff_claims`

## Wappi / draft-loop state

- Active `run_amo_wappi_draft_loop.py`: not found.
- Active watchdog screen: `56503.mango_draft_loop_watchdog` detached.
- Watchdog process cwd: `/Users/dmitrijfabarisov/Projects/Mango analyse`.
- Watchdog heartbeat: `/Users/dmitrijfabarisov/.mango_local/draft_loop/heartbeat.json`, `status=ok`, `last_cycle_at=2026-06-30T07:51:52.907023+00:00`.
- Draft-loop pairs file inventory: `/Users/dmitrijfabarisov/.mango_secrets/draft_loop_pairs.json`, 3 items, item keys only: `chat_id`, `expected_brand`, `lead_id`, `not_before_ts`, `profile_id`.
- Wappi profiles file inventory: `/Users/dmitrijfabarisov/.mango_secrets/amo_wappi_profiles.json`, 4 items, item keys only: `app_status`, `brand`, `channel`, `message_count`, `platform`, `profile_id`, `webhook_types`, `webhook_url_present`.

## Worktrees

- `/Users/dmitrijfabarisov/Projects/Mango analyse` -> `9e8fb3b14ab4687a88aa3feca4702e9e7b4f39d2`, branch `codex/tz135-direct-wow-tone`
- `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` -> `1bb51de571a54bf64765e797326024d8537661c0`, branch `codex/email-pipeline-restore`
- `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` -> `4caa5ebbe7c7411cdcb1a0e00099732da584ce00`, detached
- `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` -> `d0357d790ce75005c5d6bc06e65eb57181bd5e03`, branch `codex/adr003-semanticframe-migration`

## Secrets inventory, values not read into report

Only file metadata and key names for `.env`/top-level JSON were recorded. File content and hash fingerprints are intentionally omitted from this shared report.

| Path | Mode | Size | Mtime | Keys / shape |
| --- | --- | ---: | --- | --- |
| `~/.mango_secrets/ai_office.env` | `0600` | 101 | 2026-06-10 13:12:14 | `AI_OFFICE_API_BASE_URL`, `AI_OFFICE_API_KEY` |
| `~/.mango_secrets/amo_wappi.env` | `0600` | 673 | 2026-06-10 00:24:33 | Wappi token/login key names only |
| `~/.mango_secrets/foton_crm_readonly_mcp_connector.env` | `0600` | 146 | 2026-06-12 01:37:10 | `CONNECTOR_URL`, `BEARER_TOKEN` |
| `~/.mango_secrets/amo_wappi_phase1.json` | `0600` | 638 | 2026-06-21 13:19:05 | `allowed_test_lead_ids`, `manager_edit_log_path`, `profiles` |
| `~/.mango_secrets/amo_wappi_profiles.json` | `0600` | 1233 | 2026-06-10 00:25:39 | list[4] profile records |
| `~/.mango_secrets/draft_loop_pairs.json` | `0600` | 456 | 2026-06-12 14:57:56 | list[3] pair records |
| `~/.mango_secrets/wappi_storage_state.json` | `0600` | 9776 | 2026-06-10 00:23:07 | browser storage state keys only |

Files with less restrictive mode `0644` exist in `~/.mango_secrets` (`*.bak_tz15_*`, `draft_loop_pairs_v1_*`, `amo_wappi_profiles_v1_*`, `shared_phones_stoplist.json`, `server_access_packs/*`). Do not include their contents in shared reports; review permissions before any deploy handoff.

## Swap plan, human-run only

No swap was performed. Before any deploy:

1. Re-run freeze immediately before action:
   - `git -C /Users/dmitrijfabarisov/Projects/Mango_main_intent_ff status --short --branch`
   - `git -C /Users/dmitrijfabarisov/Projects/Mango_main_intent_ff rev-parse HEAD`
   - read heartbeat first and extract the current `pid`;
   - `ps -p <heartbeat_pid> -o pid=,ppid=,stat=,lstart=,command=`
   - `lsof -a -p <heartbeat_pid> -d cwd`
   - confirm heartbeat `status=polling`, `effective_profile=pilot_gold_v1`, `draft_path=direct_path`, and `pid` matches `ps/lsof`.
2. Run candidate smoke on the exact deploy HEAD:
   - focused pytest;
   - direct-path local smoke;
   - profile selfcheck with `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
3. Confirm Wappi draft-loop is stopped/dry-run before any live Telegram swap.
4. Stop current screen/process only after separate owner confirmation.
5. Start new process from explicitly chosen worktree and record new PID/CWD/heartbeat.
6. Validate heartbeat after start; if heartbeat/profile/snapshot mismatch, rollback.

## Rollback plan, human-run only

Current status: **DEPLOY/SWAP BLOCKED: rollback target is not set in this freeze**.

Rollback target must be recorded immediately before swap:

- previous PID/screen name;
- previous worktree path;
- previous HEAD;
- previous env/profile/snapshot;
- previous heartbeat path.

Known current live start point is insufficient as rollback target: PID `60227` runs from `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`, while its screen name points to an older `eb6fa0b` build and the worktree is now at candidate HEAD `d0357d79...`. Do not infer previous code from cwd.

Required before swap:

- `previous_worktree`: `UNSET`
- `previous_head`: `UNSET`
- `previous_screen`: `UNSET`
- `previous_command`: `UNSET`
- `previous_env/profile/snapshot`: `UNSET`
- `previous_heartbeat_path`: `UNSET`

Rollback is not automated in this freeze. Use `scripts/adr003_deploy_swap_dry_run.py` only after these previous-target fields are set. The helper intentionally does not kill processes, start live bots, edit env, or write external systems, and refuses rollback output when previous worktree equals candidate worktree unless explicitly overridden with a separately verified previous HEAD.

## Stop conditions

Stop and ask before deploy if:

- live PID/CWD/heartbeat disagree;
- profile is not `pilot_gold_v1`;
- heartbeat is stale;
- git worktree is dirty outside intended deploy artifacts;
- Wappi draft-loop is active in non-dry-run mode;
- secrets values would need to be exposed to continue;
- any smoke shows P0/brand/number/PII regression.
