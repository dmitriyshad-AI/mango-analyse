# Wappi bot-safe memory wide dry-run, 2026-06-24

Scope: wide dry-run of existing bot-safe Customer Timeline memory inside Wappi -> AMO draft flow. No AMO/Tallanto/CRM write, no Wappi send, no live-write.

Branch/worktree: `codex/wappi-botsafe-memory` at `7973901` in `/Users/dmitrijfabarisov/Projects/Mango_wappi_botsafe_memory`.

## Safety precheck

- `TELEGRAM_BOT_SAFE_CRM_CONTEXT` remains default-off: unset/`0`/`false`/empty -> disabled; `1` -> enabled.
- Customer Timeline DB used only through read-only copy: `/Users/dmitrijfabarisov/Projects/Mango_wappi_botsafe_memory/.codex_local/wappi_botsafe_memory_probe_20260624/customer_timeline_ro.sqlite`.
- Read-only copy passed `PRAGMA quick_check = ok`; connection was set to `PRAGMA query_only=ON` for the audit query.
- Wappi calls used `DefaultDenyTransport`; `mark_all=false`; no non-GET Wappi call was made by this run.
- AMO/Tallanto/CRM writes were not called. The targeted driver invoked the existing draft-loop `_process_chat_messages(..., dry_run=True)` path and stopped before any AMO-note path.
- Local raw artifacts are under ignored `.codex_local/`; they include raw ids and draft text, so they must not be committed.

## Pair inventory

Wappi list scan: 4 profiles, first pages up to 800 dialogs/profile where available.

| metric | value |
|---|---:|
| `dialogs_seen` | 2577 |
| `private_dialogs` | 2527 |
| `max_phone_seen` | 651 |
| `max_phone_stoplisted` | 4 |
| `db_paired` | 54 |
| `wappi_message_checks` | 54 |
| `with_inbound_text` | 30 |
| `memory_found` | 5 |
| `memory_empty` | 25 |

Selected for dry-run: 30 paired chats. The model run was manually stopped after 21 completed `draft_created` rows because the lower bound 20 was reached and each Codex draft call was slow. The interrupted 22nd call did not write a journal row.

Selected profile/channel split before run:

| profile/channel | dialogs | paired | selected | memory_found | memory_empty |
|---|---:|---:|---:|---:|---:|
| `foton:max` | 800 | 25 | 14 | 3 | 11 |
| `foton:telegram` | 800 | 3 | 3 | 0 | 3 |
| `unpk:max` | 177 | 26 | 13 | 2 | 11 |
| `unpk:telegram` | 800 | 0 | 0 | 0 | 0 |

Pairing reasons for non-selected/failed candidates:

| reason | count |
|---|---:|
| `contact_none_or_multi` | 60 |
| `customer_none_or_multi` | 424 |
| `lead_none_or_multi` | 128 |
| `max_phone_missing_or_stoplisted` | 328 |
| `no_inbound_text` | 24 |

## Dry-run result

| metric | value |
|---|---:|
| `draft_created` | 21 |
| `memory_found` | 4 |
| `memory_empty` | 17 |
| `pii_flagged` | 3 |
| `raw_id_flagged` | 0 |
| `foreign_brand_flagged` | 0 |
| `internal_status_flagged` | 0 |
| `url_flagged` | 2 |

By brand: `foton=14`, `unpk=7`. By channel: `max=21`; selected Telegram pairs were not reached before manual stop.

Memory usefulness summary:

| class | count | meaning |
|---|---:|---|
| `empty` | 17 | memory not injected / no brand-scoped bot-safe context |
| `partial` | 1 | memory likely moved draft into the right scenario, but weakly |
| `unclear` | 2 | memory present, but no clear effect on the draft |
| `yes` | 1 | memory clearly avoided re-asking known data |

## Per-draft table

| # | lead/chat masked | brand/channel | route | found | allowed | active_brand | items | warnings | memory_use | safety flags |
|---:|---|---|---|---:|---:|---|---:|---|---|---|
| 1 | lead `h_cd2133` / chat `h_ea07e7` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 2 | lead `h_16aa37` / chat `h_0ce3b2` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 3 | lead `h_2e3ad4` / chat `h_6b33b8` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `url:1` |
| 4 | lead `h_005d02` / chat `h_e69970` | `foton/max` | `bot_answer_self_for_pilot` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `possible_pii:possible_name` |
| 5 | lead `h_d80035` / chat `h_ab464d` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 6 | lead `h_437c50` / chat `h_f5f2c1` | `foton/max` | `bot_answer_self_for_pilot` | true | true | `foton` | 1 | `-` | `unclear` | `ok` |
| 7 | lead `h_ff8e7d` / chat `h_54c6c1` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `possible_pii:possible_name` |
| 8 | lead `h_8d2d72` / chat `h_b8d157` | `foton/max` | `draft_for_manager` | true | true | `foton` | 1 | `-` | `partial` | `ok` |
| 9 | lead `h_c9662b` / chat `h_2d57e6` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 10 | lead `h_bb01a5` / chat `h_873009` | `foton/max` | `bot_answer_self_for_pilot` | true | true | `foton` | 1 | `-` | `unclear` | `ok` |
| 11 | lead `h_e34ee8` / chat `h_cb4202` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 12 | lead `h_8eff0e` / chat `h_a949bb` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 13 | lead `h_9e47c4` / chat `h_52ee11` | `foton/max` | `bot_answer_self_for_pilot` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `possible_pii:possible_name` |
| 14 | lead `h_1d4b6c` / chat `h_e5794a` | `foton/max` | `draft_for_manager` | false | true | `foton` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `url:1` |
| 15 | lead `h_2699b6` / chat `h_c2decb` | `unpk/max` | `bot_answer_self_for_pilot` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 16 | lead `h_4511b4` / chat `h_1bee32` | `unpk/max` | `draft_for_manager` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 17 | lead `h_969234` / chat `h_78d508` | `unpk/max` | `draft_for_manager` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 18 | lead `h_51c3f1` / chat `h_338b2e` | `unpk/max` | `draft_for_manager` | true | true | `unpk` | 1 | `-` | `yes` | `ok` |
| 19 | lead `h_5fcfe2` / chat `h_71961a` | `unpk/max` | `draft_for_manager` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 20 | lead `h_9ecd7d` / chat `h_5e800b` | `unpk/max` | `bot_answer_self_for_pilot` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |
| 21 | lead `h_8a3b0f` / chat `h_e82d56` | `unpk/max` | `draft_for_manager` | false | true | `unpk` | 0 | `no_brand_scoped_bot_safe_context` | `empty` | `ok` |

## Findings

Passed checks:

- Bot-safe bridge stayed strict: `unknown` and foreign-brand chunks did not enter prompts.
- All found memory rows had `allowed_only=true`, correct `active_brand`, and `item_count=1`.
- No raw customer/timeline ids were detected in draft text.
- No internal status labels such as CRM stage names or bot-safe debug labels were detected in draft text.
- No foreign-brand text was detected by the coarse brand scan.
- Journal contains only `draft_created` with `status=dry_run`: 21 rows; no `note_written`.

Risks and notes:

- `possible_pii` in drafts #4, #7, #13: the names came from current Wappi history, not from bot-safe memory. This still matters before any live-write/send mode.
- URLs in drafts #3 and #14 need Claude review: they are not raw ids, but should be checked for customer-facing appropriateness.
- Memory was present in only 4/21 completed drafts; the strict filter is safe but low-coverage on this Wappi sample.
- Only draft #18 clearly avoided re-asking known class/subject from memory. Draft #8 was partial; #6 and #10 were unclear.
- 17/21 drafts had empty memory because no brand-scoped bot-safe chunk survived the strict filter.

## Raw artifacts

Ignored local artifacts, not for git:

- Raw private pairs: `.codex_local/wappi_botsafe_wide_dry_run_20260624/wide_pairs_private.json`.
- Raw dry-run journal: `.codex_local/wappi_botsafe_wide_dry_run_20260624/draft_loop_local_wide_targeted/journal.jsonl`.
- Masked inventory: `.codex_local/wappi_botsafe_wide_dry_run_20260624/wide_inventory_masked.json`.
- Sanitized analysis without draft text: `.codex_local/wappi_botsafe_wide_dry_run_20260624/wide_dry_run_analysis_sanitized.json`.

## Status

No production verdict. This is raw evidence for Claude #1.

`formal_pass`: read-only/dry-run plumbing stayed isolated; 21 paired Wappi chats produced draft journal rows; no external writes.

`semantic_risk`: possible names in generated drafts and low useful-memory coverage must be reviewed before any live-write memory mode.
