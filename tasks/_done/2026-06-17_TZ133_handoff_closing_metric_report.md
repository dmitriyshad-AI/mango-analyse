# TZ-133: fix honest closing metric for over-handoff

## Decision

I agree with Claude's diagnosis and used Variant A.

Reason:
- The defect is in offline measurement: `_classify_handoff_bucket` depended on `bot_close_detect.status`, but many `draft_for_manager` turns do not have it.
- Writing this status in the live layer would change live bot behavior/metadata and is outside the task.
- The safe fix is to classify closing from `client_message` in the metric layer, using the existing tone-close detector by import.

## Changed files

- `scripts/run_telegram_dynamic_client_sim.py`
- `tests/test_telegram_dynamic_client_sim.py`

Audit pack:
- `audits/_inbox/tz133_handoff_closing_metric_20260617/`

## Implementation

- Added close detection fallback through imported `_tone_close_detect_is_close_message`.
- Kept the guard: closing + explicit question does not become `closing`.
- Added `has_fabrication` and `has_hard_issue` to over-handoff candidates/examples.
- Added aggregate counters:
  - `closing_fabrication_count`
  - `closing_hard_issue_count`

No live code was changed.

## flagsON read-only validation

Input:
- `/Users/dmitrijfabarisov/Projects/Mango_measure_flags_honest/runs/flag_target_ON_ae957b6/dynamic_dialog_transcripts.jsonl`

No new run was started.

sha256 before:
- `586eb4ce5a525c0635e87ab07f5af528792ab116f397c5b00392f7629394b0c9`

sha256 after:
- `586eb4ce5a525c0635e87ab07f5af528792ab116f397c5b00392f7629394b0c9`

byte-identical:
- `true`

Metrics:
- dialogs: `31`
- turns: `90`
- handoff_turns: `27`
- over_handoff_turn_rate: `0.3`

Bucket counts:
- closing: `4`
- legitimate: `22`
- disputed_p0: `0`
- upsell_miss: `1`
- unclassified: `0`

Bucket shares:
- closing: `0.148`
- legitimate: `0.815`
- disputed_p0: `0.0`
- upsell_miss: `0.037`
- unclassified: `0.0`

Closing checks:
- `closing > 0`: yes
- closing examples with explicit `?`: `0`
- closing examples with exit markers: `0`
- `closing_fabrication_count`: `3`
- `closing_hard_issue_count`: `3`

## Tests

Targeted:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py`
- Result: `106 passed in 1.15s`

Full:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
- Result: `3326 passed, 5 skipped, 1 warning in 46.83s`

Warning:
- Existing local `urllib3` / LibreSSL warning, unrelated to TZ-133.

## Semantic review

Local semantic check: passed.

Important points:
- The metric is observe-only and does not change bot text.
- Closing no longer depends only on missing/filled `bot_close_detect.status`.
- Closing does not hide fact-audit failures.
- Raw regrade remains for Claude.

Residual note:
- The imported close detector can still treat some polite "I will wait for staff" wording as closing if there is no explicit question marker. I did not add a second detector in the metric layer, because the task explicitly required reuse of the source close detector.

## LLM calls

- `llm_calls_total`: `0`
