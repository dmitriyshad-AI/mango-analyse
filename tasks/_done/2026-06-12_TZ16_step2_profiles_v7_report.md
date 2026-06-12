# TZ-16 Step 2 Profiles V7 Report

Date: 2026-06-12

## Scope

Rebuilt customer profiles on the canonical calls DB after v7 analyze import.

No AMO/Tallanto/CRM write was executed. No ASR/R+A was launched. `stable_runtime` was read-only.

## Inputs

Source `tz12_working_batch3`:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3/`

Canonical calls DB:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`

Blacklist:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/blacklist_77.txt`

## Outputs

New ignored output folder:

`/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16/product_data/customer_profiles/tz16_profiles_v7_20260612/`

Files:

- `customer_timeline.sqlite` copied from source timeline
- `customer_profiles_micro.sqlite`
- `customer_profiles.sqlite`
- `customer_profiles_idempotence.sqlite`
- `summary.json`
- `source_hash_before.json`
- `source_hash_after.json`
- `anonymized_examples.json`

Git ignore check passed through `product_data/customer_profiles/`.

## Microprobe

- requested profiles: 5
- built profiles: 5

The microprobe completed before the full build.

## Full Build

- profiles built: 18,399
- fields written: 190,614
- superseded fields: 72,791
- full build time: 38.408 seconds
- total script elapsed: 99.028 seconds

Idempotence:

- repeated full build profiles: 18,399
- repeated full build fields: 190,614
- content signature equal: true

## Metrics Before And After

| Metric | tz12 before | tz16 after |
| --- | ---: | ---: |
| profiles | 18,399 | 18,399 |
| active fields | 118,566 | 117,823 |
| superseded fields | 70,100 | 72,791 |
| profiles with 2+ children | 4,471 | 4,411 |
| merge candidate profiles | 807 | 824 |
| merge candidate markers | 807 | 824 |

Coverage by field:

| Field | tz12 before | tz16 after |
| --- | ---: | ---: |
| parent_name | 6,185 | 6,291 |
| child_name | 6,561 | 6,623 |
| grade | 7,317 | 7,333 |
| subject | 8,935 | 8,921 |
| format | 9,148 | 9,168 |
| target_product | 8,995 | 9,023 |
| next_step | 9,895 | 9,941 |
| objection | 11,269 | 11,300 |
| child_slot_merge_candidate | 807 | 824 |
| tallanto_balance | 0 | 0 |
| tallanto_group | 0 | 0 |
| payment_fact | 0 | 0 |
| amo_stage | 0 | 0 |
| amo_status | 0 | 0 |

## Analyze Version Counts

- analysis done with valid JSON: 65,939
- v7 summaries: 22,679
- non-v7 summaries: 43,260
- blacklist IDs loaded: 77
- blacklist IDs present in master: 77
- blacklist IDs with v7: 0
- blacklist IDs preserved old: 77

The 77 blacklist rows are expected to remain on old summaries by Dmitry's decision.

## Source Hash Guard

`tz12_working_batch3` hash before and after matched:

`source_tz12_unchanged=true`

The builder used a copied timeline DB from the new output folder, not the source timeline path.

## Anonymized Examples

Stored in:

`product_data/customer_profiles/tz16_profiles_v7_20260612/anonymized_examples.json`

Five examples include only profile hashes, field names, counts, brands, source systems, phone presence, child slot counts, and value lengths. They do not include names, raw phones, emails, message text, or source identifiers.

## Tests

Targeted tests:

`tests/test_tz16_profiles_v7_build.py tests/test_customer_profile_builder.py tests/test_refresh_customer_profiles.py tests/test_tz12_pii_gitignore.py`

Result:

`24 passed`

Compile check:

`python3 -m py_compile scripts/build_tz16_profiles_v7.py` passed.

## NEG

- Old `tz12_working_batch3` hash unchanged before/after.
- New raw artifacts are under ignored `product_data/customer_profiles/`.
- Repeat full build has identical content signature.
- Anonymized examples do not expose raw values.
- Source timeline was copied before build; source folder was not used as output.

## LLM Calls

`llm_calls_total`: 0 for Step 2.

## Status

`formal_pass`: yes.

`semantic_pass`: `PASS_WITH_NOTES`.

The profile layer is rebuilt for analysis and review. It is not a live CRM writeback and not permission to start Stage B contact-card writes.
