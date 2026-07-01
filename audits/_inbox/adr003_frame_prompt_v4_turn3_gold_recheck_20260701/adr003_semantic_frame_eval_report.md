# ADR-003 SemanticFrame Eval Report

- Acceptance: `pass`
- Technical shadow status: `pass`
- Semantic decision status: `not_pass`
- Active behavior allowed: `False`
- ON turns: `241`
- Frame present: `241` / `241`
- Frame schema complete: `241` / `241`
- OFF/ON route-text diffs: `0`
- OFF/ON input diffs: `0`
- LLM call mode: `semantic_frame_enrichment`
- LLM raw total delta: `-296`
- LLM expected extra calls: `241`
- LLM non-frame ON calls: `0`
- Frame decision shadow turns: `241`

## Acceptance Flags

- `extra_model_calls_expected`: `True`
- `hard_gate_failures_zero`: `True`
- `input_turns_match`: `True`
- `route_text_diff_zero`: `True`
- `semantic_frame_present_on_all_turns`: `True`
- `semantic_frame_required_fields_complete`: `True`

## Notes

- ON run is paired SemanticFrame enrichment; model calls are only post-hoc frame metadata calls.
