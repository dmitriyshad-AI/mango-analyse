# TZ-118 Block D Segment Guard Shadow Report

Generated: 2026-06-16 03:09 MSK

## Scope

- Block: D, mono-call role assignment.
- Mode: default `off`; measured `shadow` with `MONO_ROLE_SEGMENT_GUARD_MODE=repair`.
- Primary/writeback: not enabled.
- Model transport: Codex CLI only, no `OPENAI_API_KEY`.
- Live systems: no AMO/Tallanto/CRM writes, no ASR, no DB writes.
- Design file note: `2026-06-15_RAZBOR_i_dizayn_gr1_gr4.md` was not found locally in this worktree or main project tree; implementation follows Dmitry's chat specification.

## Changed Files

- `src/mango_mvp/config.py`
- `src/mango_mvp/services/transcribe.py`
- `scripts/run_tz116_mono_role_gold50_measure.py`
- `scripts/build_tz117_error_traces.py`
- `tests/test_dialogue_format.py`
- `tests/test_smoke.py`
- `tests/test_tz116_offline_modes.py`

## Implementation

- Added `MONO_ROLE_SEGMENT_GUARD_MODE`, default `off`.
- Added deterministic manager anchors:
  - greeting/presentation with center marker;
  - request for class/name/phone/email/child data;
  - manager-side payment/link offer.
- Added segment guard for runs where Codex roles disagree with rule roles for at least 3 consecutive turns.
- Added short service-turn handling through the same guard metadata.
- Preserved `raw_model_roles_before_guard` and `post_guard_roles` so measurements can compare before/after.
- Extended TZ-117 D trace with raw model, post-guard model, guarded/changed flags, and guard effect type.

## Shadow Artifacts

- Shadow run: `audits/_inbox/tz118_d_segment_guard_shadow_20260616_025521/`
- Summary: `audits/_inbox/tz118_d_segment_guard_shadow_20260616_025521/summary.json`
- Variant probe: `audits/_inbox/tz118_d_segment_guard_shadow_20260616_025521/variant_probe.json`
- Trace: `audits/_inbox/tz118_d_segment_guard_trace_20260616_030753/`
- Trace summary: `audits/_inbox/tz118_d_segment_guard_trace_20260616_030753/tz117_trace_summary.json`

All artifacts are under ignored `audits/_inbox/`.

## Measurement

Input: 23 gold-labeled mono calls, 924 turns.

Raw Codex before guard:

- Errors: 55 / 924
- Error rate: 5.95%
- Exact calls: 4 / 23
- Mean per-turn accuracy: 94.62%

After `segment_guard=repair`:

- Errors: 220 / 924
- Error rate: 23.81%
- Exact calls: 1 / 23
- Mean per-turn accuracy: 76.35%
- Guarded turns: 254
- Changed turns: 199
- Guard fixed: 17
- Guard broke: 182
- Guard net delta: -165

Error segments:

- Raw segment-turn ratio: 5.95%
- Post-guard segment-turn ratio: 23.81%
- Raw error segments length >= 3: 5
- Post-guard error segments length >= 3: 41

Trace effect counts:

- `no_guard`: 670
- `broke`: 182
- `neutral_correct`: 52
- `fixed`: 17
- `neutral_wrong`: 3

## Variant Probe Without New Model Calls

Replayed raw Codex roles from the same run:

- `anchor_only`: 55 errors, net 0, changed 0.
- `low_info_plus_anchor`: 67 errors, net -12.
- `run_anchor_homogeneous_manager`: 67 errors, net -12.
- `run_homogeneous`: 67 errors, net -12.

Interpretation: manager anchors are already classified correctly by Codex in this gold set; moving short/service or segment runs back to the weak rule creates regressions.

## Stop Conditions

`summary.json` marks:

- `post_guard_worse_than_raw=true`
- `guard_net_delta_negative=true`
- `guard_broke_total_positive=true`
- `stop_recommended_for_primary=true`

Semantic verdict: no semantic pass for D primary. The segment guard should not be promoted. The current finding is a negative regrede candidate: useful as evidence that the proposed deterministic repair is unsafe on this gold set.

## Tests

- Targeted: `27 passed, 1 warning`
- Full: `3299 passed, 5 skipped, 1 warning`

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_format.py tests/test_tz116_offline_modes.py::test_mono_role_gold50_measure_calls_codex_only_for_low_confidence tests/test_tz116_offline_modes.py::test_mono_role_gold50_measure_reports_segment_guard_net_effect tests/test_tz116_offline_modes.py::test_tz117_d_trace_marks_low_info_rationale tests/test_tz116_offline_modes.py::test_tz117_d_trace_marks_segment_guard_effect tests/test_smoke.py::SmokePipelineTest::test_get_settings_parses_float_env_values
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

## Safety

- `llm_calls_total=23`, all via Codex CLI role assignment shadow.
- No OpenAI API key used.
- No CRM/Tallanto/AMO writes.
- No ASR.
- No stable runtime writes.
- B/E/A/C not started.

## Recommendation

Stop Block D here for Claude/Dmitry regrede. Do not flip primary. Next design should avoid handing long disagreement runs to the current rule engine; if a deterministic guard is kept, it should probably be mark-only or limited to evidence stronger than the current rule.
