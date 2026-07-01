# ADR-003 SemanticFrame Report Script

Date: 2026-07-01
Branch: `codex/adr003-semanticframe-migration`

## What changed

- Added `scripts/report_adr003_semantic_frame_eval.py`.
- Added tests in `tests/test_report_adr003_semantic_frame_eval.py`.
- Documented OFF/ON report command in `docs/ADR003_SEMANTIC_FRAME_EVAL.md`.

## What the report checks

- OFF/ON route/text/safety/checklist diffs by dialog id and turn.
- OFF/ON LLM call delta from `dynamic_summary.json`.
- SemanticFrame coverage and required-field completeness.
- `must_handoff` alignment against actual route handoff and P0/high-risk signal.
- `frame_decision_shadow` status and mismatch examples.

## Safety

- The script only reads saved dynamic simulator artifacts and writes a local report.
- Missing OFF artifacts or missing OFF/ON summaries cannot produce `pass`; they force `needs_review`.
- No live bot process was touched.
- No AMO/Tallanto/CRM writes.
- No runtime behavior changed.

## Auditor fix

Independent audit found that missing OFF artifacts could be treated as a successful flag. Fixed before commit: `route_text_diff_zero` now requires a real compared OFF/ON pair with no missing turns, and `extra_model_calls_zero` now requires a measured zero delta.
