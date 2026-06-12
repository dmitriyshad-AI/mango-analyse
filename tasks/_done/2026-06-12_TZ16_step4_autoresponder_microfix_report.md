# TZ-16 Step 4: anti-autoresponder microfix

Date: 2026-06-12
Branch: `codex/tz16-d4-profiles-v7-rerun-tail`

## Scope

Fixed the v7 analysis path so long live MANAGER/CLIENT dialogue is not collapsed into `non_conversation` only because the transcript contains words such as `абонент`, `секретарь`, third-party company names, `перезвонить`, or IVR-like menu fragments.

No DB writes, no AMO/Tallanto/CRM writes, no ASR, no Resolve+Analyze.

## Changed files

- `src/mango_mvp/services/analyze.py`
- `src/mango_mvp/quality/non_conversation.py`
- `tests/test_analyze.py`
- `tests/test_non_conversation_quality.py`

## Implementation

- v7/full prompt now explicitly forbids using `non_conversation` for long or multi-turn dialogue unless the client side is exclusively system/IVR/voicemail/no-live text.
- `detect_non_conversation_signals` now protects long third-party business dialogues: the transcript can mention a collector/company/menu, but if both sides have substantial turns and human/business-response markers, it becomes manual live-context, not forced `non_conversation`.
- `_detect_call_type` maps guarded borderline-live cases to `service_call`, so they are not blocked before the LLM step.
- Existing hard blocks remain for true voicemail, virtual secretary and short IVR.

## Real read-only controls

Blacklist sample checked: `12617`, `14115`, `14327`, `15112`, `16146`.

After the fix:

- `5/5` are not forced by guardrails as `non_conversation`.
- `5/5` deterministic `call_type = service_call`.
- Labels: all `manual_review_borderline_live_context`.

True autoresponder controls checked: `15717`, `16565`, `24790`.

After the fix:

- `3/3` remain `non_conversation`.
- `3/3` remain high-confidence forced no-live / autoresponder.

## LLM microprobe

Allowed Step 4 microprobe was run through `codex_cli`, model `gpt-5.4-mini`, profile `full`.

Sanitized output path, ignored by git:

- `product_data/customer_profiles/tz16_profiles_v7_20260612/step4_blacklist_microprobe.json`

Counters:

- `llm_calls_total = 5`
- elapsed seconds: `89.688`
- `5/5` normalized to `service_call`
- `5/5` prompt version `v7`
- `0/5` guardrail forced `non_conversation`
- `0/5` raw transcripts written to artifact
- `0/5` raw summaries written to artifact

Per-call sanitized result:

| call_id | call_type | guardrail_label | guardrail_force | needs_review | target_product_present | next_step_present |
|---:|---|---|---:|---:|---:|---:|
| 12617 | service_call | contentful_protected_live_dialogue | false | false | true | true |
| 14115 | service_call | contentful_protected_live_dialogue | false | false | false | true |
| 14327 | service_call | manual_review_borderline_live_context | false | false | false | true |
| 15112 | service_call | manual_review_borderline_live_context | false | false | false | true |
| 16146 | service_call | manual_review_borderline_live_context | false | false | false | true |

Semantic note: the microprobe fixed the target class (`false autoresponder`) on all 5 calls, but one case still has `target_product_present=true`. This is not a blocker for the microfix, but it is a reason not to run the full 77-call rerun without a small reviewed batch or an additional semantic check.

## Tests

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_non_conversation_quality.py tests/test_analyze.py -k "non_conversation or autoresponder or prompt or ivr or voicemail or third_party"` -> `35 passed, 1 skipped, 22 deselected`
- `python3 -m py_compile src/mango_mvp/quality/non_conversation.py src/mango_mvp/services/analyze.py` -> passed

## NEG

- Synthetic long third-party business dialogue is not forced as IVR.
- `_detect_call_type` keeps that dialogue as `service_call`.
- Pure short IVR still forces `non_conversation`.
- Virtual secretary / voicemail tests still pass.
- Real canonical autoresponder controls `15717`, `16565`, `24790` still force `non_conversation`.

## Safety

- No raw transcripts or raw summaries in git.
- Microprobe artifact is under ignored `product_data/customer_profiles/`.
- No external writes.
- Full 77-call rerun was not started.

## LLM calls

`llm_calls_total = 5` for role `analyze_microprobe`.
