# ADR-003 F2ab: local current-head re-enrich report

Дата: 2026-07-02
Ветка: `codex/adr003-semanticframe-migration`
Код: `a3019910` (`ae4843a6` + doc-only bundle report)

## Что было проверено

По просьбе Дмитрия замер сделан локально, без передачи на M1.

Использован безопасный режим `--semantic-frame-enrich-from`: старые OFF-транскрипты не пересоздавались, а текущий код пересчитал только posthoc `SemanticFrame` + proof/reconciliation/self-answer shadows.

Это не трогает:

- живой Telegram-бот;
- Telegram polling;
- AMO/CRM;
- Tallanto;
- профиль и live-флаги.

Live pid `60227` не трогался.

## Вход

OFF source:

`audits/_inbox/adr003_f2_m1_9a57be1_rehydrated_36ea110_20260702/OFF/dynamic_dialog_transcripts.jsonl`

Local output:

`audits/_inbox/adr003_f2ab_local_reenrich_20260702173307/`

SHA manifest:

`audits/_inbox/adr003_f2ab_local_reenrich_20260702173307/sha_manifest.json`

## Safety result

Full local re-enrich:

- dialogs: `131`
- turns: `241`
- hard_gate_failures: `0`
- script result: `ok=true`

Eval acceptance:

- `route_text_diff_zero=true`
- `input_turns_match=true`
- `semantic_frame_present_on_all_turns=true`
- `semantic_frame_required_fields_complete=true`
- report acceptance: `pass`

Gold calibration:

- compared rows: `77`
- `must_handoff_accuracy=0.8831`
- `too_confident=0`
- `too_cautious=9`
- `p0_misses=0`
- report acceptance: `needs_review`

## What changed vs 36ea110 diagnosis

The prompt calibration helped the narrow confusion class:

- previous F2y on 36ea110: `frame_action_blocks_proof=1`;
- local current-head re-enrich: `frame_action_blocks_proof=0`;
- action counterfactual: `scope_confusion_total=0`.

So the specific "existence/course format vs live availability" action confusion improved in this fixed-input re-enrich measurement.

## Why active autonomy is still NO-GO

Over-handoff report:

- `safe_handoff_total=10`
- `safe_draft_for_manager=3`
- `safe_manager_only=7`
- `draft_candidates_for_future_active=0`
- `harmless_context_ack_status_candidates=0`

Frame calibration queue:

- `strict_active_candidates_now=0`
- `fact_gated_strict_f3_draft_candidates=0`
- `current_safe_over_handoff=10`
- active readiness: `no_go`

Current handoff fact gap:

- `current_handoff_rows=4`
- `route_only_candidates=0`
- `danger_excluded=2`
- `proof_axis_mismatch=2`
- `frame_action_blocks_proof=0`

Partial-answer report:

- `partial_support_handoff_turns=40`
- `draft_partial_shadow_candidates=2`
- `manager_only_partial_policy_blocked=3`
- `hard_missing_axis_blocked=5`
- `action_or_danger_excluded_partial_rows=30`
- active readiness: `no_go`

Action counterfactual:

- `cases=4`
- `new_active_candidates=0`
- `safe_reference_counterfactual_exact_proof_total=4`
- `counterfactual_residual_hard_missing_total=2`
- `negative_controls_preserved_total=2`
- active readiness: `no_go`

## Two partial candidates found

They are useful for future design, not for immediate enablement:

1. `wappi_pair_missing_72h_002#1`, client excerpt: "Расскажите про обе"
   - route: `draft_for_manager`
   - frame: `risk_class=missing_facts`, `answerability=manager_only`, `must_handoff=true`
   - partial support exists, but requires owner-approved partial-answer text policy.

2. `p0_model_led_pos_how_next#1`, client excerpt: "Подскажите, как дальше: сначала тестирование или сразу группа? Я не понимаю порядок."
   - route: `draft_for_manager`
   - frame: `risk_class=missing_facts`, `answerability=manager_only`, `must_handoff=true`
   - partial support exists, but requires owner-approved partial-answer text policy.

Neither is approval to send text.

## Interpretation

This is a `formal_pass` for local shadow telemetry and a `NO-GO` for active Ф3.

What is proven:

- current prompt calibration improved the specific `requested_action=check_availability` confusion class on fixed transcripts;
- safety gates stayed clean in local re-enrich;
- no P0 was lowered;
- there are still zero strict active candidates.

What is not proven:

- this is not a full fresh bot-run on current direct-path text generation;
- it does not prove customer-safe partial-answer text;
- it does not justify profile/live enablement.

Next useful work:

- design a narrow partial-answer text policy for safe `draft_for_manager` rows with proven facts;
- keep it shadow-only first;
- require semantic review before any active route/text change.
