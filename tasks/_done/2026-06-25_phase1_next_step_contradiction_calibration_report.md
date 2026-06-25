# Phase 1 next_step contradiction calibration

Date: 2026-06-25

Branch: `codex/phase1-dossier-enrich`

## Scope

Fixed false `next_step` defers in bot-safe Phase 1 memory on a test copy of Customer Timeline.

The original issue: a later event after a closed stage was treated as a contradiction and rendered as "Уточнить у менеджера". That is a normal follow-up pattern, not a data conflict.

No AMO, Tallanto, CRM, stable runtime, or production Customer Timeline writes were performed.

## Code Changes

- `next_step_resolver`: "contradiction -> manager" is now limited to real data conflicts:
  - explicit conflict markers in the source text;
  - structured brand conflict;
  - structured grade/class conflict.
- Later questions or later events after closed stages no longer force manager review by themselves.
- Broad false-positive marker `ошиб` was removed from negation checks because it matched educational phrases like "ошибки в оформлении".
- `bot_safe_summary`: exact-detail scrub now removes standalone times like `10:00`.
- `post_layers`: bot-safe memory step guard now also blocks unsupported "next step" instructions when memory says the step is empty or needs manager review, including top-level `bot_safe_context_items` used by the replay runner.

## Test-Copy Rebuild

Input DB:

`runs/phase1_ON_micro_input/customer_timeline_phase1_testcopy.sqlite`

Output DB:

`runs/phase1_next_step_fix_v3_input/customer_timeline_phase1_nextstep_fix_v3.sqlite`

`bot_safe_summary` next-step statuses:

| Status | Before | After |
|---|---:|---:|
| `active` | 2569 | 3177 |
| `closed` | 2009 | 3749 |
| `empty` | 10595 | 10595 |
| `needs_manager_review` | 2828 | 480 |

Exact time leak check in bot-safe summary:

| Check | Before | After |
|---|---:|---:|
| summaries containing `10:00` | 50 | 0 |

## Phase-1 ON Micro Replay

Same 20-dialog Phase-1 memory-rich replay set.

Before:

- dialogs: 20
- verdicts: `PASS=1`, `PASS_WITH_NOTES=10`, `FAIL=9`
- violated gates: `fabrication=9`
- LLM calls: `152` total (`memory=66`, `bot_draft=66`, `judge=20`)

After:

- output: `runs/phase1_next_step_fix_v5_ON`
- dialogs: 20
- verdicts: `PASS=2`, `PASS_WITH_NOTES=18`, `FAIL=0`
- violated gates: none
- LLM calls: `152` total (`memory=66`, `bot_draft=66`, `judge=20`)

## Tests

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_next_step_resolver.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_bot_safe_memory_step_guard.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_bot_safe_memory_step_guard.py tests/test_customer_timeline_next_step_resolver.py tests/test_customer_timeline_bot_safe_summary.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline* tests/test_*bot_safe* tests/test_*direct_path*
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Results:

- targeted next-step resolver: `20 passed`
- bot-safe memory step guard: `10 passed`
- combined targeted: `47 passed`
- requested Phase-1 target set: `320 passed`
- full pytest: `3621 passed, 5 skipped, 1 warning`

## Semantic Status

Formal pass: yes.

Semantic pass: `PASS_WITH_NOTES` for the 20-dialog micro replay. The false contradiction class is corrected on this set, and no hard safety failures remained. This is not a production verdict.

Residual risks:

- model variability can still produce different wording on repeated runs;
- subject conflicts are intentionally not treated as automatic structured conflicts because a client can legitimately discuss multiple subjects;
- broader Phase-1/M1 replay remains the next gate before enabling memory.

## Boundaries

- AMO writes: `0`
- Tallanto writes: `0`
- CRM writes: `0`
- client sends: `0`
- production Customer Timeline writes: `0`
- stable runtime changes: `0`

Verdict "ready for prod" was not made.
