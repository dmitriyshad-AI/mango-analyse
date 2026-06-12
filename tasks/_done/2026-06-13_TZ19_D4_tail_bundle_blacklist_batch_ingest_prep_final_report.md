# TZ-19 Final Report: tail bundle, blacklist batch, ingest prep

Дата: 2026-06-13  
Ветка: `codex/tz19-d4-tail-bundle-blacklist-ingest`

## Commits

- `87cc2d2` - TZ19 block A add M1 tail bundle builder
- `5f024ed` - TZ19 block C add tail import dry-run script
- `11b72ff` - TZ19 block B report blacklist batch15

## Block A: M1 Tail Bundle

Bundle:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_tail_20260612/`

M1 task:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1/2026-06-13_analyze_tail_20260612_d4_codex.task.yaml`

Ready marker:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1/2026-06-13_analyze_tail_20260612_d4_codex.task.yaml.ready`

Bundle counters:

- selected calls: `3439`
- transcript chars: `15682102`
- duration sec: `598435`
- blacklist overlap: `0`
- parts: `860/860/860/859`
- ids sha256: `8680b5456824ac7159cc1ec5993399aa8ae57712602aa4d4c2d582b65041ad5e`
- prompt version: `v7`
- prompt sha256: `12718ea6b8a5ee500910300c4c2de7c3695f78217c3b63a62d572de612b5eacf`
- slice DB bytes: `116289536`
- slice DB sha256: `83df6dafdad644135a7d73f84419410d7c2eb4e0e969f90217a5263eb62b4ce0`
- code archive commit: `87cc2d2cd3977a8ac94242bee7295babdcbde3cc`
- code archive sha256: `3e21cfbb552129c277a2370d8f8d7f11fb487c0173acbf2ddf38e19b1b2b0564`

Acceptance checks:

- manifest count equals required `3439`;
- `ids_all.txt` has `3439` unique IDs;
- `slice_zone.db` has `3439` rows and `PRAGMA quick_check=ok`;
- prompt sha in manifest equals sha of `prompt/analyze_prompt_full_v7.txt`;
- `.ready` contains sha256 of task yaml.

## Block B: Blacklist Batch 15

Ignored artifact root:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz19_blacklist_batch15/`

Full summaries for review:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz19_blacklist_batch15/results_batch15.jsonl.gz`

Selected calls:

`16628`, `19055`, `19871`, `21767`, `22259`, `24772`, `26435`, `27025`, `27191`, `28320`, `28911`, `29433`, `34882`, `35059`, `37544`

Excluded TZ-16 microprobe calls:

`12617`, `14115`, `14327`, `15112`, `16146`

Run counters:

- total: `15`
- done: `15`
- failed: `0`
- pending: `0`
- prompt version: `v7` for `15/15`
- model: `gpt-5.4-mini` for `15/15`
- elapsed sec: `196.385`
- `llm_calls_total=15`

15-row table is in:

`tasks/_done/2026-06-13_TZ19_blockB_blacklist_batch15_report.md`

Summary:

- `service_call` after: `8`
- `non_conversation` after: `7`
- target product present after: `6`

Semantic status: `PASS_WITH_NOTES`. Full 77 blacklist rerun remains gated by review of `results_batch15.jsonl.gz`.

## Block C: Tail Import Script

Script:

`scripts/import_tz19_analyze_tail_results.py`

Tests:

`tests/test_tz19_analyze_tail_import.py`

Behavior:

- default mode is dry-run;
- `--apply` requires `--backup-to`;
- writes only `analysis_json`, `analysis_status`, `analysis_json_chars`, `has_analysis_json`, `last_error`;
- rejects rows outside manifest whitelist;
- rejects blacklist rows;
- rejects duplicate conflicts;
- rejects bad JSON, wrong schema, wrong prompt version/model and transcript mismatch;
- idempotent repeated import of same payload.

Real import of `analyze_tail_20260612` was not run.

## NEG

- Bundle selector did not include blacklist: overlap `0`.
- Bundle selector stopped unless count is exactly `3439`.
- Rebuilding bundle is idempotent for generated files.
- Batch15 excludes TZ-16 microprobe IDs.
- Batch15 made exactly `15` LLM calls.
- True autoresponder controls `15717`, `16565`, `24790` remain deterministic `non_conversation_high_confidence` without LLM.
- Import script dry-run does not write.
- Import script `--apply` requires backup path.
- Repeated import of identical synthetic results gives `0` updates.
- Out-of-manifest and blacklist rows are rejected.
- Whitelist columns are enforced by one explicit SQL update.

## Tests

Targeted:

```text
12 passed, 1 warning in 0.39s
```

Full pytest:

```text
3097 passed, 2 skipped, 1 warning in 39.26s
```

## Audit Pack

`audits/_inbox/tz19_tail_bundle_blacklist_ingest_20260613/`

Contains:

- `implementation_notes.md`
- `changed_files.txt`
- `test_output.txt`
- `semantic_review.md`
- `risk_review.md`
- `backward_compatibility.md`

## Git Status Note

Tracked TZ-19 files are committed through the block commits above. The working tree still has pre-existing unrelated untracked handoff/prompt files, left untouched:

- `D1_audit_backlog/existing_clients/HANDOFF_live_card_v1_to_chief_architect_2026-06-12.md`
- `tasks/_inbox_codex/2026-06-12_TZ16_merge_and_M1_tail_bundle_PROMPT_for_D4.md`
- `tasks/_inbox_codex/2026-06-12_TZ19_D4_tail_bundle_blacklist_batch_ingest_prep.md`
- `tasks/_inbox_codex/2026-06-12_TZ19_PROMPT_for_D4.md`
