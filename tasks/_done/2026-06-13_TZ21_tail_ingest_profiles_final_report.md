# TZ-21 final report: tail 3,439 import and profile rebuild

Дата: 2026-06-13

## Commits

- Block A import report: `5b31096 TZ21 block A import tail results`
- Block B profile rebuild/code/tests/audit: `58e376d TZ21 block B rebuild profiles after tail import`

## Block A: canonical import

Input:

- `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_tail_20260612/results_part1..4.jsonl.gz`
- Rows: 3,439
- ids sha256: `8680b5456824ac7159cc1ec5993399aa8ae57712602aa4d4c2d582b65041ad5e`
- prompt sha256: `12718ea6b8a5ee500910300c4c2de7c3695f78217c3b63a62d572de612b5eacf`
- blacklist overlap: 0

Backup before write:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_20260613`

Idempotence-run backup:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db.backup_before_tz21_tail_idempotence_20260613`

Counters:

- Dry-run: `read=3439`, `updated=3439`, `rejected=0`.
- Apply: `read=3439`, `updated=3439`, `rejected=0`.
- Repeat apply: `read=3439`, `updated=0`, `skipped_same=3439`, `rejected=0`.
- DB `quick_check`: `ok`.
- Total v7 rows: `22679 -> 26118`.
- Manifest rows with v7 after import: `3439`.
- Blacklist rows with v7: `0`.

## Block B: profile rebuild

Output folder:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz21_profiles_after_tail_20260613/`

Key local artifacts:

- `summary.json`
- `tz21_comparison.json`
- `rerun_tail_after_tz21.json`
- `customer_profiles.sqlite`
- `customer_profiles_idempotence.sqlite`

These files are ignored by git.

Build counters:

- Microprobe: 5 profiles, 109 fields.
- Full build: 18,399 profiles, 190,670 fields.
- Full build time: 36.576 sec.
- Total script elapsed: 95.982 sec.
- Unmatched calls during build: 1,053.
- Idempotence: `content_signature_equal=true`.
- Source `tz12_working_batch3` unchanged: `true`.

Profile metrics before `tz16_profiles_v7` -> after TZ-21:

- Profiles: `18399 -> 18399`.
- Fields total: `190614 -> 190670`.
- Active fields: `117823 -> 117702`.
- Superseded fields: `72791 -> 72968`.
- Profiles with 2+ children: `4411 -> 4410`.
- Merge-candidate profiles: `824 -> 830`.
- Merge-candidate markers: `835 -> 841`.

Coverage deltas by key field:

- `child_name`: `6623 -> 6631`.
- `grade`: `7333 -> 7336`.
- `format`: `9168 -> 9172`.
- `parent_name`: `6291 -> 6294`.
- `next_step`: `9941 -> 9940`.
- `objection`: `11300 -> 11297`.
- `target_product`: `9023 -> 9008`.
- `subject`: `8921 -> 8921`.
- `tallanto_balance`, `tallanto_group`, `payment_fact`, `amo_stage`, `amo_status`: still `0`, as expected; no new CRM/Tallanto source was added in this block.

Zone metrics:

- Zone calls current v7: `16797 -> 20236`, delta `+3439`.
- Zone calls current not-v7: `29390 -> 25951`, delta `-3439`.
- Long non-blacklist old-summary tail after TZ-21: `0`.
- Long old-summary including blacklist after TZ-21: `56`.
- Blacklist ids in zone: `56`; preserved as old summaries.

## Anonymized profile examples

- `profile_example_1`: hash `bac8255934dc`, phone present, events `409`, active fields `0`, child slots `0`.
- `profile_example_2`: hash `55d79fc10eb7`, phone present, events `349`, active fields `38`, child slots `7`, brand `foton`, source `mango_processed_summary`.
- `profile_example_3`: hash `62a3c3862801`, phone present, events `313`, active fields `7`, child slots `1`, brand `unknown`, source `mango_processed_summary`.
- `profile_example_4`: hash `e29dcec3c150`, phone present, events `261`, active fields `6`, child slots `2`, sources `customer_profile_builder`, `mango_processed_summary`.
- `profile_example_5`: hash `4121d88682dc`, no phone, events `246`, active fields `0`, child slots `0`.

No raw names, phones, transcripts, quotes, or profile values are included in this report.

## Tests and gates

- Targeted pytest: `24 passed in 0.60s`.
- Full pytest: `3100 passed, 2 skipped, 1 warning in 45.34s`.
- Audit pack: `/Users/dmitrijfabarisov/Projects/Mango analyse/audits/_inbox/tz21_tail_ingest_profiles_20260613/`
- Semantic review: `PASS_WITH_NOTES`.

## Safety

- AMO/CRM write: no.
- Tallanto write: no.
- Email/Telegram send: no.
- ASR: no.
- Resolve+Analyze: no.
- Blacklist import: no.
- LLM calls total: `0`.

## Notes

- During Block B, the profile build initially exposed a path-handling bug in read-only SQLite opening under the project path with a space. Fixed by using `mode=ro&immutable=1` and covered with regression tests.
- Active field count decreased by 121 while total field count increased by 56. This is plausible after replacing older summaries with stricter v7 summaries and superseded rules, but downstream CRM-card review should watch it.
- Some profiles with many events still have zero active fields; that is a data coverage issue, not a TZ-21 import failure.
