# Refactor subscription_llm merge acceptance — 2026-06-11

## Scope

Architect verdict for `codex/refactor-subscription-llm`: PASS.

Task: merge the refactor branch into canonical `main`, run full pytest, then run a live smoke18 regression on the main Mac. Cleanup of old worktrees/branches is explicitly gated by architect confirmation of the smoke result.

No AMO/Wappi/Tallanto/Telegram live writes were run.

## Merge

Main folder before merge:

- path: `/Users/dmitrijfabarisov/Projects/Mango analyse`
- branch: `main`
- head: `61fb34ca`

Command:

```bash
git merge --ff-only codex/refactor-subscription-llm
```

Result:

- fast-forward succeeded;
- new `main` head after merge: `8402b776`;
- no conflict resolution was performed.

Git output summary:

- `30 files changed`
- `69817 insertions(+)`
- `13537 deletions(-)`
- refactor split created `src/mango_mvp/channels/subscription_llm_parts/*`
- facade remains `src/mango_mvp/channels/subscription_llm.py`

## Full pytest

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Result:

```text
3007 passed, 2 skipped, 1 warning in 45.28s
```

Warning:

- local Python `urllib3` / LibreSSL compatibility warning.

## Smoke18

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
CODEX_HOME=/private/tmp/mango_codex_home_memory_profile_20260611_fast \
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl \
  --snapshot product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json \
  --bot-mode codex \
  --memory-mode codex \
  --memory-reasoning low \
  --semantic-mode codex \
  --semantic-reasoning medium \
  --parallel 4 \
  --judge-prompt-version v9.1 \
  --out-dir runs/20260611_refactor_merge_smoke18
```

Output directory:

- `runs/20260611_refactor_merge_smoke18`

Top-line result:

```json
{
  "dialogs": 18,
  "turns": 41,
  "pass": 10,
  "pass_with_notes": 8,
  "fail": 0,
  "hard_gate_failures": 0,
  "ok": true
}
```

Validity checks from `dynamic_summary.json`:

- `config_validity.invalid=false`
- `run_config.key_flags.profile.effective=true`
- `run_config.key_flags.render.effective=true`
- `run_config.key_flags.rubric.effective=true`
- `run_config.key_flags.retriever.effective=true`
- `run_config.key_flags.memory_provenance.effective=true`
- `llm_calls.bot_direct_draft=39`
- `llm_calls.bot_retriever=39`
- `llm_calls.bot_semantic_output_verifier=42`
- `llm_calls.bot_faithfulness=0`
- `llm_calls.memory=0`

P0 branch flags:

- `p0_branch_hits=4`
- `direct_path_preblocked_p0=2`
- `payment_dispute_manager_only=2`

## Cleanup status

Cleanup was not executed yet.

Reason: task says to clean old worktrees and delete the merged branch only after architect confirmation of the smoke. Smoke artifacts are ready for architect review.

Still present:

- `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor`
- `/Users/dmitrijfabarisov/Projects/Mango analyse.tz13-main`
- `/Users/dmitrijfabarisov/Projects/Mango analyse.wave1-probe`
- local branch `codex/refactor-subscription-llm`

Planned after architect OK:

1. Move untracked `runs/` from `.tz13-main` and `.wave1-probe` to `runs/_from_worktrees/` in the main folder.
2. Remove `.tz13-main`, `.wave1-probe`, `.refactor`.
3. Delete merged branch `codex/refactor-subscription-llm`.
4. Run `git worktree prune`.
5. Confirm final git status.

## Status

Formal acceptance checks passed:

- fast-forward merge succeeded;
- full pytest green;
- smoke18 completed with `FAIL=0` and `hard_gate_failures=0`.

Semantic/regression verdict over smoke transcripts remains with the architect.

