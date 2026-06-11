# Gold pack v2 build and replay pair — 2026-06-11

## Scope

Built `real_manager_gold_v2_2026-06-11.yaml` from:

- `D1_audit_backlog/gold_v2_intake/gold_pack_v2_review_2026-06-10.docx`
- `D1_audit_backlog/GOLD_v2_owner_fixes_2026-06-11.md`

Owner fixes were applied for examples 2, 4, and 15. The v1 pack remains the default unless `TELEGRAM_BOT_GOLD_REAL_PACK` points to v2.

## Artifacts

- `product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/real_manager_gold_v2_2026-06-11.yaml`
- `D1_audit_backlog/GOLD_v2_owner_fixes_2026-06-11.md`
- r4 queue entries in `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3_sources/release_manifest.yaml`

Pack contents:

- 38 examples.
- Examples 2/4/15 are taken from owner fixes, not the docx source text.
- `manager_response_masked` has placeholders for numbers/dates/addresses; prompt examples are rendered without square-bracket tokens.
- `source` is preserved for every example.

## r4 Queue

Added internal-only `manual_decision_fact_overrides`, not client-safe facts:

- `gold_v2_r4_queue.foton.midyear_new_student_curator_support`
- `gold_v2_r4_queue.unpk.midyear_new_student_curator_support`
- `gold_v2_r4_queue.foton.small_groups_personal_attention`
- `gold_v2_r4_queue.unpk.small_groups_personal_attention`

All four are `status: needs_owner_confirmation`, `route_policy: manager_handoff_only`, `internal_only: true`, with empty `client_safe_text`.

## Code

- `TELEGRAM_BOT_GOLD_REAL_PACK` now changes the loaded pack path.
- `direct_path.gold_pack_version` reflects the active pack stem when examples are used.
- Default behavior remains v1 when the env override is absent.

## Tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "real_manager_gold"
5 passed, 444 deselected
```

Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3009 passed, 2 skipped, 1 warning
```

## Replay Pair

Replay source:

- `runs/20260610_rubric_base_smoke89/dynamic_dialog_transcripts.jsonl`

Runs:

- OFF v1: `runs/20260611_gold_v2_replay89_OFF_v1pack`
- ON v2: `runs/20260611_gold_v2_replay89_ON_v2pack`

Both runs were valid:

- `config_validity.invalid=false`
- `bot_direct_draft=292`
- `bot_retriever=292`
- `bot_faithfulness=0`
- `bot_semantic_output_verifier>0`

Summary:

| Metric | OFF v1 | ON v2 |
| --- | ---: | ---: |
| dialogs | 89 | 89 |
| turns | 323 | 323 |
| FAIL | 8 | 8 |
| PASS | 22 | 19 |
| PASS_WITH_NOTES | 59 | 62 |
| hard_gate_failures | 8 | 8 |
| tone_score | 62.7 | 62.2 |
| tone_warm | 184 | 176 |
| tone_canc | 5 | 8 |
| non-P0/self tone_score | 66.0 | 65.2 |
| over_handoff_turn_rate | 0.625 | 0.638 |
| handoff_turns | 202 | 206 |
| semantic verifier downgrade_rate | 0.0433 | 0.0217 |
| llm_calls total | 1114 | 1119 |

Hard failure sets:

OFF v1:

- `sm_f_p0_complaint`
- `sm_u_xbrand1`
- `sm_u_xbrand2`
- `sm_u_camp1`
- `sm_f_camp_cancel`
- `sm_f_trial`
- `sm_u_docs_close`
- `sm_f_return_client`

ON v2:

- `sm_f_p0_complaint`
- `sm_u_night_camp`
- `sm_u_camp1`
- `sm_u_camp_zvsh`
- `sm_f_camp_pay`
- `sm_f_discount_press`
- `sm_f_trial`
- `sm_u_docs_close`

Resolved FAILs:

- `sm_f_camp_cancel`
- `sm_f_return_client`
- `sm_u_xbrand1`
- `sm_u_xbrand2`

New FAILs:

- `sm_f_camp_pay` — judge: `fabrication`; rationale cites PII echo of child name.
- `sm_f_discount_press` — judge: `internal_leak`; rationale cites PII echo of child name.
- `sm_u_camp_zvsh` — judge: `made_a_promise`; rationale says “Записала” before enough data / confirmed waitlist action.
- `sm_u_night_camp` — judge: `internal_leak`; rationale cites PII echo of client name.

## Semantic Review

Status:

- `formal_pass`: yes.
- `semantic_pass`: no for enabling v2 by default.
- `pilot_ready`: no until architect reviews raw transcripts and the new FAILs are either fixed or classified as measurement issues.

What passed:

- Schema and lint checks.
- Default v1 behavior remains intact.
- v2 can be selected explicitly by env override.
- r4-dependent facts were not made client-safe.

Blocking issues before enabling v2:

- Criteria “0 new hard failures” is not met.
- Tone metric slightly drops.
- Over-handoff rate slightly rises.
- Example 15 depends on r4 confirmation and must not be treated as fully grounded before r4.

Recommended next action:

- Keep default pack at v1.
- Give both replay folders to the architect for raw review.
- Treat v2 as an experimental pack only until the four new FAILs and the r4 fact coverage are resolved.
