# TZ-16 Step 3: rerun tail sizing

Date: 2026-06-12
Branch: `codex/tz16-d4-profiles-v7-rerun-tail`

## Scope

Read-only sizing of the current customer rerun zone after v7 import and profile rebuild.

No AMO/Tallanto/CRM writes, no ASR, no Resolve+Analyze, no LLM calls.

## Files

- Script: `scripts/compute_tz16_rerun_tail.py`
- Tests: `tests/test_tz16_rerun_tail.py`
- Ignored aggregate output: `product_data/customer_profiles/tz16_profiles_v7_20260612/rerun_tail_report.json`

## Source formula

The TZ-13 rerun zone was reproduced from `customer_timeline.sqlite`:

- active AMO customers: `1303`
- strong Tallanto student customers: `7298`
- union customers: `7965`
- calls in zone: `46187`
- calls missing in canonical master DB: `0`

Formula:

- AMO active = `customer_opportunities.source_system='amocrm_snapshot'`, `opportunity_type='amo_deal'`, empty `closed_at`, excluding statuses `Закрыто и не реализовано`, `Успешно`.
- Tallanto strong = `identity_links.source_system='tallanto_snapshot'`, `link_type='tallanto_student_id'`, `match_class='strong_unique'`.
- Zone = union of these customers, Mango calls from `timeline_events`.

## v7 coverage

- first-slice ids total: `22756`
- first-slice ids inside customer zone: `16853`
- customer-zone calls currently on v7: `16797`
- customer-zone first-slice blacklist preserved old: `56`
- customer-zone calls still on old summaries: `29390`

## Tail breakdown

Old summaries inside customer zone:

- total old-summary calls: `29390`
- transcript chars: `31796935`
- non-blacklist old-summary calls: `29334`
- non-blacklist transcript chars: `31135982`
- old `>=60` sec calls including blacklist: `3495`
- old `>=60` sec transcript chars including blacklist: `16343055`
- old `>=60` sec calls excluding blacklist: `3439`
- old `>=60` sec transcript chars excluding blacklist: `15682102`

Reasons among old-summary calls:

- blacklist preserved: `56`
- before `2025-06-01`: `8585`
- below `60` sec: `20749`
- eligible not in first slice under the original filter: `0`
- targeted but not current v7: `0`
- not done / empty transcript: `0`

Reason quadrants:

- recent `>=2025-06-01` and `>=60` sec, blacklist: `56`, transcript chars `660953`
- recent `>=2025-06-01` and `>=60` sec, non-blacklist: `0`, transcript chars `0`
- recent `>=2025-06-01` and `<60` sec: `20749`, transcript chars `12637285`
- old `<2025-06-01` and `>=60` sec: `3439`, transcript chars `15682102`
- old `<2025-06-01` and `<60` sec: `5146`, transcript chars `2816595`

By duration:

- `0-14` sec: `9539`
- `15-29` sec: `7245`
- `30-59` sec: `9111`
- `60-119` sec: `1548`
- `120-299` sec: `1468`
- `300-599` sec: `410`
- `600+` sec: `69`

By recency:

- `0-30` days: `263`
- `31-90` days: `2405`
- `91-180` days: `3468`
- `181-365` days: `14391`
- `366+` days: `8863`
- unknown date: `0`

## Estimate

Same filter as first slice (`started_at >= 2025-06-01`, duration `>=60`, done, transcript present, excluding blacklist):

- remaining calls: `0`
- estimated time: `0`

All non-blacklist old tail:

- calls: `29334`
- transcript chars: `31135982`
- rough serial estimate by call count: `161.116` hours
- rough 4-way wall estimate by call count: `40.279` hours
- rough wall estimate by observed M1 throughput: `40.494` hours
- rough serial estimate by transcript chars: `32.371` hours
- rough 4-way wall estimate by transcript chars: `8.093` hours

The char-based estimate is more representative for this tail because most remaining calls are short.

Old `>=60` sec tail including blacklist:

- calls: `3495`
- transcript chars: `16343055`
- rough 4-way wall estimate by call count: `4.799` hours
- rough wall estimate by observed M1 throughput: `4.825` hours
- rough 4-way wall estimate by transcript chars: `4.248` hours

Old `>=60` sec tail excluding blacklist:

- calls: `3439`
- transcript chars: `15682102`
- rough 4-way wall estimate by call count: `4.722` hours
- rough wall estimate by observed M1 throughput: `4.747` hours
- rough 4-way wall estimate by transcript chars: `4.076` hours

Practical note: do not rerun the `56` blacklist calls until the anti-autoresponder prompt fix is reviewed. The most useful next batch is old `>=60` sec excluding blacklist; short calls should be a separate decision because they dominate call count but carry low text volume.

## Checks

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz16_rerun_tail.py` -> `4 passed`
- `python3 -m py_compile scripts/compute_tz16_rerun_tail.py` -> passed
- `git check-ignore -v product_data/customer_profiles/tz16_profiles_v7_20260612/rerun_tail_report.json` -> ignored by `product_data/customer_profiles/`
- Secret/PII grep over script, test and aggregate report -> no matches

## NEG

- closed / won AMO statuses do not enter active AMO zone.
- ambiguous Tallanto student links do not enter strong Tallanto zone.
- blacklist calls are counted separately and are not included in same-filter second slice.
- output contains aggregate counts only, no raw phones, emails or names.

## LLM calls

`llm_calls_total = 0`
