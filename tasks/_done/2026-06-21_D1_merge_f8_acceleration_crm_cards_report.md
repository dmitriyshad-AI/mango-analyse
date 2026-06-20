# D1 merge: F8 -> accelerations -> CRM cards

Date: 2026-06-21
Worktree: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`
Base before merges: `0e0c7b7`
Final local main: `5480bbd`
Pushed: no
Live bot / AMO / Tallanto / CRM writes: no

## Read-only merge map

TZ source: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_vlivanie_3_vetok_v_main.md`

`git pull --ff-only` was not possible because local `main` and `origin/main` had diverged:

- local `main`: `0e0c7b7 customer-profile: record combined cache rerun`
- `origin/main`: `43134ae TZ139: expose customer profile card projection fields`

The TZ explicitly referenced local `main 0e0c7b7` as the starting point, so merges were performed on the clean local `main` worktree. No push was made.

Dry merge checks:

- `codex/f8-axes-catalog-selector`: `merge-tree rc=0`, no overlapping files with local main.
- `codex/tz-uskoreniya-3-punkta`: `merge-tree rc=0`; only overlap with local main was `docs/worktrees_registry.md`.
- `codex/etap1-crm-card-assembler`: `merge-tree rc=0`; only overlap with local main was `docs/worktrees_registry.md`.
- Between accelerations and CRM cards the real shared files were:
  - `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
  - `tests/test_customer_timeline_canonical_readonly_import.py`

## Merges

1. `4b5ba30 Merge F8 price axes catalog selector`
   - Branch: `codex/f8-axes-catalog-selector` at `d0da5ba`
   - Clean merge.

2. `2df1cf5 Merge customer timeline acceleration package`
   - Branch: `codex/tz-uskoreniya-3-punkta` at `c0c76ea`
   - Clean merge.
   - `docs/worktrees_registry.md` auto-merged by git.

3. `5480bbd Merge CRM card assembler`
   - Branch: `codex/etap1-crm-card-assembler` at `a8428c0`
   - Clean merge.
   - `canonical_readonly_import.py` and its test auto-merged by git.
   - Post-merge grep confirmed both parts are present:
     - `with store.bulk_write()`
     - `canonical_calls_db`
     - `read_calls_by_phone`
     - `call_event(...)`

## Tests

After F8:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `3374 passed, 5 skipped, 1 warning`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline*.py`
  - `163 passed`

After accelerations:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `3410 passed, 5 skipped, 1 warning`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline*.py`
  - `190 passed`

After CRM cards:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `3428 passed, 5 skipped, 1 warning`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline*.py`
  - `190 passed`

## Determinism checks

Bulk-write customer timeline check on temporary DBs:

- `logical_hash_equal: true`
- final hash: `8bcf6e0030552841505143c09ce2014bb17d243ea3fe3e16f0b9e78d36184b9f`

Parallel analysis migration check on temporary DBs:

- sequential workers: `1`, rows: `40`, code: `0`
- parallel workers: `4`, rows: `40`, code: `0`
- `rows_equal: true`
- hash: `501232388cf5787f9760b54c2e8f1d5e3b58c867b1216fba3fa61e22e0d2e71b`

## CRM card preview

Accepted preview:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_llm_history_r2/`

Rebuilt preview in `/tmp` using the accepted history-summary cache:

`/tmp/mango_crm_preview_merge_3O4c`

Comparison:

- CSV identical byte-for-byte: `true`
- CSV sha256: `36ad32b71c119cbf6ddaa23149016b28ac9a7984867c79fd5417cc1917203488`
- Summary differs only in cache accounting:
  - accepted: `cache_hits=9`, `cache_misses=1`, `llm_calls=1`
  - rebuilt: `cache_hits=10`, `cache_misses=0`, `llm_calls=0`

## F8 flag / YAML / live

- `TELEGRAM_PRICE_AXES_SELECTOR` env is unset.
- `TELEGRAM_PRICE_AXES_SELECTOR` is not in `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- F8 was not enabled.
- No source YAML files changed in these merge commits.
- No live bot, AMO, Tallanto, CRM write, ASR, or Resolve+Analyze was run.
- No push was made.
