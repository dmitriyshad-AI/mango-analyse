# TZ-20 blacklist-57 finish report

Дата: 2026-06-14

## Scope

Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tz20_blacklist`

Branch: `codex/tz20-blacklist57`

Выполнено по ТЗ-20:

- Блок А: локальный прогон оставшихся 57 blacklist-звонков на `gpt-5.4-mini`, full v7.
- Блок Б: код режима `--blacklist-override` + dry-run вливания, без реального `--apply`.

Запрещённые действия не выполнялись: CRM/AMO/Tallanto write, ASR, M1, профильная пересборка, реальное вливание в canonical DB.

## Block A: Remaining 57

Source blacklist:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/blacklist_77.txt`

Счётчики:

- blacklist total: `77`
- TZ16 microprobe excluded: `5`
- TZ19 batch15 excluded: `15`
- remaining local run: `57`
- duplicate ids: `0`
- controls `15717`, `16565`, `24790`: не входят в blacklist-77 и проверялись отдельно без LLM.

Remaining 57:

```text
38370, 41101, 42057, 42659, 42980, 46547, 47860, 49031, 51275, 51910,
52992, 53079, 53100, 57873, 58340, 60692, 61646, 61961, 62005, 62170,
62218, 62250, 62423, 62441, 62505, 62518, 62535, 62654, 62750, 62856,
63060, 63178, 63265, 63275, 63289, 63293, 63341, 63649, 63926, 64021,
64128, 64245, 64322, 64382, 64428, 64443, 64519, 64603, 64605, 64615,
64622, 64654, 64666, 64671, 64763, 64781, 64786
```

Run:

- runner: `scripts/run_analyze_ab_test.py`
- source DB: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/data/slice_zone.db`
- arm: `mini_v7:gpt-5.4-mini:full`
- prompt sha256: `12718ea6b8a5ee500910300c4c2de7c3695f78217c3b63a62d572de612b5eacf`
- elapsed: `1228.175` sec
- `llm_calls_total_current_task=57`

Runner metrics:

- total: `57`
- done: `57`
- failed: `0`
- pending: `0`
- v7/mini meta rows: `57`
- marked_non_conversation: `11`
- summary_missing: `0`
- summary_looks_like_dialogue_dump: `0`
- summary_contains_english: `0`
- next_step_contains_english: `1` (`email`, false-positive metric)

Ignored artifact root:

`product_data/customer_profiles/tz20_blacklist77_results/`

Key artifacts:

- `results_remaining57.jsonl.gz`
- `results_import_ready72.jsonl.gz`
- `results_blacklist77_for_import_dryrun.jsonl.gz`
- `manifest_blacklist77.json`
- `review_table_77.csv`
- `summary_77.json`
- `neg_autoresponder_controls.json`

Unified 77 review aggregates:

- payload sources: `tz16_sanitized_microprobe=5`, `tz19_batch15=15`, `tz20_remaining57=57`
- import-ready rows: `72`
- sanitized/no import payload rows: `5`
- call_type after: `service_call=59`, `non_conversation=18`
- target_product_present after: `46`
- needs_review after: `18`
- non_conversation + needs_review: `18`

Important note: the 5 TZ16 microprobe calls were previously saved only as a sanitized metrics artifact without full `analysis_json`. They are included in the 77-row review table, but not import-ready.

NEG controls:

- `15717`: deterministic `non_conversation`, no LLM
- `16565`: deterministic `non_conversation`, no LLM
- `24790`: deterministic `non_conversation`, no LLM

## Block B: Import dry-run

Changed importer:

- `scripts/import_tz19_analyze_tail_results.py`

New mode:

- `--blacklist-override <ids-file>`
- without override, blacklist manifest still fails with `manifest intersects blacklist`
- override file must contain only blacklist ids and only ids present in manifest
- non-blacklist result rows are rejected in override mode
- counters now expose `accepted_needs_review`, `accepted_non_conversation`, `accepted_non_conversation_needs_review`

All-77 dry-run:

- mode: `dry_run`
- effective_allowed_ids: `77`
- read: `77`
- updated: `72` (`would_update`; no factual DB write in dry-run)
- accepted_for_import: `72`
- accepted_needs_review: `18`
- accepted_non_conversation: `18`
- accepted_non_conversation_needs_review: `18`
- rejected_not_done: `5` (`12617`, `14115`, `14327`, `15112`, `16146`)
- other rejects: `0`
- before prompt_version_rows: `0`
- after prompt_version_rows: `0`

Import-ready 72 dry-run:

- mode: `dry_run`
- effective_allowed_ids: `72`
- read: `72`
- updated: `72` (`would_update`; no factual DB write in dry-run)
- accepted_needs_review: `18`
- accepted_non_conversation_needs_review: `18`
- rejects: `0`

No-override NEG:

- command fails with `manifest intersects blacklist`
- return code: `2`

No real `--apply` was run.

## Tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q \
  tests/test_tz19_analyze_tail_import.py \
  tests/test_tz19_tail_bundle.py \
  tests/test_tz19_calls_review_table.py
```

Result: `24 passed, 1 warning`.

Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q
```

Result: `3109 passed, 5 skipped, 1 warning in 46.25s`.

Full output:

`audits/_inbox/tz20_blacklist57_20260614/full_pytest_output.txt`

## NEG Coverage

- blacklist manifest without override is blocked
- override ids must be blacklist ids
- override ids must exist in manifest
- non-blacklist rows cannot pass through blacklist override
- blacklist rows missing from explicit override are rejected
- `needs_review` and `non_conversation+needs_review` counters are preserved
- dry-run is stable and does not write
- apply path on synthetic DB updates only whitelisted analysis columns and requires backup

## Semantic Status

`formal_pass`: code tests and full pytest are green; local run completed 57/57; no real import was applied.

`semantic_pass`: `PASS_WITH_NOTES`.

Notes:

- The current task produced exactly 57 new LLM calls.
- For later live import, 72 rows are import-ready now.
- The 5 TZ16 microprobe rows need a separate decision before canonical import: either recover full original payloads, rerun those 5 with explicit approval, or accept that only 72/77 can be imported from existing artifacts.
- The 5 TZ16 microprobe rows are excluded from `ids_import_ready72.txt` and do not have import payloads.
- `non_conversation+needs_review` rows are deliberately counted separately and must not be treated as fully resolved without review.
- The 18 `non_conversation+needs_review` rows can be imported only as explicitly marked review rows, not as normal client dialogues.
- Independent review was a formal/contract review of counters, safety and metadata. Raw `analysis_json` and raw transcript text were not semantically audited in full.

## Changed Files

- `scripts/import_tz19_analyze_tail_results.py`
- `tests/test_tz19_analyze_tail_import.py`
- `tasks/_done/2026-06-14_TZ20_blacklist57_finish_report.md`

## Next Gate

Architect/Claude should review:

- `product_data/customer_profiles/tz20_blacklist77_results/review_table_77.csv`
- `product_data/customer_profiles/tz20_blacklist77_results/results_remaining57.jsonl.gz`
- `product_data/customer_profiles/tz20_blacklist77_results/results_import_ready72.jsonl.gz`
- `product_data/customer_profiles/tz20_blacklist77_results/import_dryrun_all77_report.json`
- `product_data/customer_profiles/tz20_blacklist77_results/import_dryrun_import_ready72_report.json`

Real `--apply` remains blocked until separate approval after regрейд.
