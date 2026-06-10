# TZ12 D4 Delivery 2 Report

Date: 2026-06-10
Branch: `codex/tz12-d4-client-history-profile`

## Scope

Completed RP-3..RP-7 for D4 existing client history profile:

- RP-3: deterministic `customer_profile` store, contracts, builder and CLI.
- RP-4: Telegram export import into customer timeline.
- RP-5: WhatsApp export import into customer timeline.
- RP-6: incremental profile refresh by timeline `--since` and copied journal quiet detection.
- RP-7: CRM summary preview without CRM writes.
- Extra TZ12 hardening from real runs: malformed channel rows, SQLite path/WAL read-only lookup, batch timeline writes, WhatsApp phone dedupe, master calls without phones.

## Commits

- `9bbbd03b` TZ12 RP3 build deterministic customer profiles
- `b1ab0bf5` TZ12 RP4 import Telegram history to timeline
- `33a1a537` TZ12 RP5 import WhatsApp history to timeline
- `4b5a84f9` TZ12 RP6 refresh customer profiles incrementally
- `11f0e972` TZ12 RP7 preview CRM customer summaries
- `91b9e698` TZ12 RP4 harden Telegram export import
- `518ca9d1` TZ12 RP5 treat malformed WhatsApp fragments as warnings
- `8e65e642` TZ12 batch timeline import writes
- `df1b833c` TZ12 RP3 skip master calls without phones
- `f9fbf483` TZ12 RP5 link WhatsApp phone chats to existing customers

Delivery 1 was reported separately in `tasks/_done/2026-06-10_TZ12_D4_delivery1_report.md`.

## Real Batch Artifacts

All generated local data is git-ignored:

- `product_data/customer_profiles/tz12_working_batch3/customer_timeline.sqlite`
- `product_data/customer_profiles/tz12_working_batch3/customer_profiles.sqlite`
- `product_data/customer_profiles/tz12_working_batch3/telegram_import_report.json`
- `product_data/customer_profiles/tz12_working_batch3/whatsapp_import_report.json`
- `product_data/customer_profiles/tz12_working_batch3/profile_build_report.json`

Audit pack:

- `audits/_inbox/tz12_delivery2_2026-06-10/`

## Import Counters

Telegram:

- dialogs: 600
- messages: 7,708
- imported: 6,950
- linked_by_phone: 2,120
- session_only: 4,830
- groups skipped: 175
- skipped total: 758
- validation_ok: true

WhatsApp:

- chats: 4,620
- messages_seen: 63,584
- imported records: 40,034
- linked_by_phone: 39,999
- session_only: 35
- existing unique phone matches: 2,693
- service skipped: 7,416
- empty skipped: 16,134
- malformed warning: 1
- validation_ok: true, status: `completed_with_warnings`

Timeline after channel imports:

- customer_identities: 18,399
- channel events: 46,984
- Telegram events: 6,950
- WhatsApp events: 40,034

## Profile Build

- profiles_built: 18,399
- fields_written: 188,666
- superseded_fields: 70,100
- ambiguous_calls: 0
- unmatched_calls: 1,053
- build time: 16.45 seconds

Most frequent fields:

- next_step: 35,664
- objection: 35,630
- subject: 24,086
- target_product: 23,898
- format: 22,832
- child_name: 17,740
- parent_name: 14,603
- grade: 14,213

## Examples

Five anonymized profile examples and three anonymized CRM summary examples:

- `audits/_inbox/tz12_delivery2_2026-06-10/anonymized_examples.md`

Examples intentionally omit raw values, full phones, names, emails, message text and service identifiers.

## Tests

Full pytest:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
- result: `2946 passed, 2 skipped, 1 warning in 44.53s`
- output: `audits/_inbox/tz12_delivery2_2026-06-10/full_pytest_output.txt`

NEG / regression coverage includes:

- no password/raw PII git tracking for TZ12 profile artifacts;
- `analysis_meta` only for fresh analyze, not migration/export backfill;
- Telegram malformed rows skipped;
- Telegram path-with-spaces and WAL read-only lookup;
- Telegram group skip, outbound direction, null phone, repeat dedupe;
- WhatsApp malformed fragment warning;
- WhatsApp phone chat links to existing customer without duplicate profile;
- WhatsApp dry-run does not create timeline DB;
- idempotent WhatsApp re-run;
- customer_profile duplicate handling and superseded fields;
- master call without phone counted as unmatched;
- ambiguous phone in builder not auto-merged;
- incremental refresh by `created_at`;
- copied journal quiet detection, unmatched and ambiguous quiet pairs;
- CRM preview masks phones, filters service/legal fields and does not auto-render first profile when several profiles match one phone.

## Semantic Review

Independent semantic review: `PASS_WITH_NOTES`, no blocking issues.

Notes:

- Detailed Telegram operation report under `product_data/customer_profiles/tz12_working_batch3/` contains internal source refs and must stay out of git/public audit packs.
- Importers do not yet expose explicit `ambiguous_phone_matches`; current behavior avoids unsafe first-match merge, but can leave separate channel profiles.
- Telegram malformed test covers JSON objects with missing fields, not invalid JSONL syntax.

Semantic review file:

- `audits/_inbox/tz12_delivery2_2026-06-10/semantic_review.md`

## LLM Calls

Delivery 2 LLM calls: 0.

Delivery 1 A/B runner used LLM only in RP-2 on existing transcripts:

- official matrix: 200 calls total (`mini_v6`, `mini_v7`, `gpt54_v6`, `gpt55_v6`: 50 each)
- plus one successful pilot call before the official matrix

No LLM calls were made for Telegram/WhatsApp imports, profile build, refresh, or CRM preview.

## Safety Boundaries

No writes to AMO, Tallanto, Wappi, CRM, external channels, `channels/*`, `draft_loop`, `amo_wappi_*`, `pilot_context_assembly.py`, `pilot_gold_v1`, or `stable_runtime` DB/audio/transcripts.

`stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db` was used read-only for profile enrichment.

Global git status is not clean because the worktree contains pre-existing unrelated modified/untracked files from parallel tracks. TZ12 tracked changes are committed separately; generated/raw artifacts are ignored.
