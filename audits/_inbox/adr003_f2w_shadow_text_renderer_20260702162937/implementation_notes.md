# Implementation Notes

## Scope

Implemented a report-only shadow text renderer readiness layer in:

- `scripts/report_adr003_frame_calibration_queue.py`
- `tests/test_report_adr003_frame_calibration_queue.py`

The layer is measurement-only. It does not change runtime behavior, provider code, direct path, route, text, profile, or live bot settings.

## What The Renderer Does

For proof-reconciliation rows, the report now computes:

- `shadow_text_renderer_status`
- `shadow_text_renderer_blockers`
- `shadow_text_renderer_source`
- `shadow_text_candidate_length`
- `shadow_text_candidate_hash`
- `shadow_text_candidate_exported=false`

The full candidate text is intentionally not exported.

## Safety Scope

The only currently renderable shape is an atomic class-range fact:

- `fact_type=course_parameter`
- `structured_value.classes_raw` or `structured_value.classes`
- fresh, client-safe, correct brand, no PII, no template requirement

Everything else remains blocked.

## 36ea110 Result

- `proof_reconciliation_would_reconcile=9`
- `proof_text_shadow_renderer_candidates=0`
- `shadow_text_renderer_by_status={blocked_wrong_brand: 5, blocked_unsupported_structured_value: 3, blocked_template_renderer_not_implemented: 1}`

This confirms no active text/route step is ready.
