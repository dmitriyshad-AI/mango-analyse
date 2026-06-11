# TZ-13 + TZ-8 merge to main

Date: 2026-06-11
Feature branch: `codex/preserve-wave6-profile-final-dirty`
Target branch: `main`
Base before rebase: `6800755a`

## Included work

Accepted original commits were rebased on current `main`:

| Original | Rebased | Scope |
|---|---|---|
| `525f3128` | `80769881` | Judge v9.1 calibrations and `--replay-from` |
| `eff6f5f7` | `5bdb7022` | TZ-13 replay checks report |
| `12d796bf` | `0cb09191` | `TELEGRAM_MEMORY_PROVENANCE` memory slots with provenance, default OFF |
| `bf815a8e` | `d52f59eb` | Judge v9.1 strict P0 first-turn handling |

## Pre-merge checks

- `TELEGRAM_MEMORY_PROVENANCE` is not present in `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- Targeted NEG:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py tests/test_dialogue_memory.py tests/test_draft_loop.py -k 'judge_v91 or replay_from or memory_provenance or build_memory_model or draft_loop_persists'
13 passed, 134 deselected
```

- Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
2980 passed, 5 skipped, 1 warning
```

## Runtime notes

- Default judge prompt version for new runs is now `v9.1`.
- Comparisons across runs are valid only inside one judge version. If old runs used `v9`/`v2`, re-judge both sides with the same version before comparing.
- `--replay-from` is for replaying the same client inputs through the current bot/judge path; it is not a substitute for a fresh dynamic client run when client behavior may change.
- `TELEGRAM_MEMORY_PROVENANCE` remains default OFF and is not part of `pilot_gold_v1`; paired measurement is still required before profile enablement.
