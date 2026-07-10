# ADR003 final M1 pachka 99b8169a 2026-07-07

- Scenario: `adr003_final_m1_pachka_99b8169a_20260707.jsonl`
- Lines: 250 = 2 spec + 248 personas
- Contract: `ON` is current `pilot_gold_v1` profile on HEAD; `B` emulates old profile with explicit `=0` overrides and old reading/apply lists.
- Runner: `scripts/run_adr003_final_pachka_pair.sh`

## Leg Contract

`ON`:
- no manual target env;
- `TELEGRAM_SEMANTIC_READING_CLASSES` unset;
- `TELEGRAM_READING_APPLY_CLASSES` unset;
- target flags unset so profile defaults apply.

`B`:
- `TELEGRAM_FACT_SELECT_FRAME=0`
- `TELEGRAM_TONE_CLOSE_FRAME_VETO=0`
- `TELEGRAM_P0_MODEL_LED=0`
- `TELEGRAM_PROSE_MODEL_LED=0`
- `TELEGRAM_PAYMENT_REFUND_DISPUTE_SPLIT=0`
- `TELEGRAM_SEATS_DEFAULT_OPEN=0`
- `TELEGRAM_P0_LATCH_AUTORELEASE_V2=0`
- `TELEGRAM_SEMANTIC_READING_CLASSES=sense_seats,slots_gsf,off_topic,intent_actions,live_status_read`
- `TELEGRAM_READING_APPLY_CLASSES=live_status_read/conversation_intent_plan`

Validator:
- `ON` requires trace classes: `fact_select_read,route_templates,reask_read,roles_read`.
- `B` forbids the same trace classes.

## Composition

- Canonical ADR003 E2 set: 156 personas.
- Combo/fact-select focus with calibrated `29a/29b`: unique personas from `adr003_kombo_factsel_veto_masker_24922645_20260707`.
- D-043 focus set: unique personas from `adr003_focus_reask_roles_payment_20260706`.
- Custom fixtures:
  - 4 latch/autorelease multi-turn controls;
  - 5 seats/default-open/floor controls;
  - 3 seasonal ambiguity controls;
  - 12 slots background controls without `slots-1b`.

## Mini-Smoke Fact Select

Local mini-smoke on HEAD `99b8169a`:
- OUT: `runs/adr003_fact_select_minismoke_99b8169a_20260707_122708`
- Report: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-07_FACT_SELECT_minismoke_99b8169a_report.md`
- Result: keep `fact_select` in package.
- Counts: `applied=4`, `fail_closed=4`; fail-closed reasons were `empty_product` or `low_confidence`.

## Dry Check

Local runner dry-check:
- OUT: `runs/adr003_final_pachka_dry_99b8169a_123748`
- `VALID_E3_ON`: dialogs=2 turns=2 frames=2 eligible_frame_rate=1.0000, required trace classes present.
- `VALID_E3_B`: dialogs=2 turns=2 frames=2 eligible_frame_rate=1.0000, forbidden trace classes absent.
