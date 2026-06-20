# D1 F8 clean defer fix on local main

Date: 2026-06-21
Worktree: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`
Base: `1c7e6b4`
Code fix commit: `d5d08db Fix F8 price axes clean defer`
Source fix reference: `d20a7b58 Fix F8 price-axis clean defer`
Pushed: no
Live bot / AMO / Tallanto / CRM writes: no

## Applied delta

Only the three product files requested from `d20a7b58` were applied:

- `src/mango_mvp/channels/fact_retrieval.py`
- `src/mango_mvp/knowledge_base/price_axes_catalog.py`
- `tests/test_kb_price_axes_catalog.py`

Stat:

```text
3 files changed, 105 insertions(+), 5 deletions(-)
```

The audit/report files from `d20a7b58` were not cherry-picked into this local main fix.

## What changed

- Added explicit weekday/weekend schedule axis for the F8 price selector.
- Added default-off flag `TELEGRAM_PRICE_AXES_CLEAN_DEFER`.
- When both `TELEGRAM_PRICE_AXES_SELECTOR=1` and `TELEGRAM_PRICE_AXES_CLEAN_DEFER=1`, a dead-end price query can return an empty fact pack instead of pulling unrelated facts.
- Preserved valid price selection, including the УНПК 5th-grade weekend semester case.

## Flags

Both F8 flags remain default OFF:

- `TELEGRAM_PRICE_AXES_SELECTOR`: not in `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`
- `TELEGRAM_PRICE_AXES_CLEAN_DEFER`: not in `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`

Runtime env during direct-path smoke explicitly set both to `0`.

## Tests

Targeted F8 tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_price_axes_catalog.py
13 passed in 1.25s
```

Full pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3432 passed, 5 skipped, 1 warning in 55.46s
```

Customer timeline tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline*.py
190 passed in 2.64s
```

Note: the current repo has more than the historical 11 `test_customer_timeline_*` tests; I ran the full current glob.

## Direct-path smoke

Command shape:

```text
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
TELEGRAM_PRICE_AXES_SELECTOR=0
TELEGRAM_PRICE_AXES_CLEAN_DEFER=0
PYTHONPATH=src python3 scripts/run_telegram_dynamic_client_sim.py
  --scenarios product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl
  --snapshot product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json
  --out-dir runs/20260621_f8_clean_defer_direct_smoke
  --limit 1 --max-turns 1 --parallel 1
  --client-mode fake --judge-mode fake --bot-mode codex
  --memory-mode off --semantic-mode fake --semantic-verifier-mode fake
  --judge-prompt-version v9.1 --timeout-sec 180
```

Result:

- out dir: `runs/20260621_f8_clean_defer_direct_smoke`
- dialogs: `1`
- turns: `1`
- verdict: `PASS_WITH_NOTES` from fake judge
- fail: `0`
- hard gate failures: `0`
- `config_validity.invalid=false`
- direct path attempted: `true`
- direct path model called: `true`
- `llm_retrieve.used=true`
- retrieval mode: `id_only`
- fallback reasons: none

The smoke intentionally used fake client/judge/memory/verifier to verify the live direct-path bot branch without live writes or a full semantic run.

## Safety

- No push.
- No live bot restart.
- No AMO/Tallanto/CRM writes.
- No source YAML changes.
- No F8 profile enablement.
