# ADR-003 M1 eval adapter

Date: 2026-07-01
Branch: `codex/adr003-semanticframe-migration`

## What changed

- Added `--client-mode scripted` to `scripts/run_telegram_dynamic_client_sim.py`.
- Added `initial_history_lines` support in the dynamic simulator context seed.
- Exposed `bot_semantic_frame` and `bot_frame_decision_shadow` in `dynamic_turns.csv`.
- Sanitized generated ADR-003 Wappi/M1 files for raw `@` fragments and likely FIO.
- Removed the current Wappi client message from seeded history even when an auto-reply follows it.
- Extended `scripts/build_adr003_semantic_frame_eval.py` to produce a runner-compatible M1 scenario bundle:
  - `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_m1_scenarios_20260701.jsonl`
  - `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_eval_manifest_20260701.json`
  - `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_baseline_20260701.json`
  - `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_wappi_latest25_20260701.jsonl`
- The M1 bundle has 131 personas:
  - 25 sanitized Wappi pair-missing what-if cases.
  - 106 existing persona regressions from forward-payment, P0, riskzone, reliable, closing, and tz147 sets.

## Safety

- No live bot process was touched.
- No AMO/Tallanto/CRM writes.
- No route/text behavior flag was enabled.
- Changes are limited to offline eval/simulator infrastructure and tracked eval data.
- Generated `adr003_*20260701*.jsonl` files have no raw `@` characters after sanitization.
- Final generated files were rechecked for the Wappi names caught by auditors (`Ирина`, `Ольга`, `Фирсов`, `Буличев`, `Ярослав`, `Сергей`) and all marker counts are 0.
- Both the Wappi latest25 case file and the runnable M1 scenario remove the current client message from seeded history; this prevents the evaluation from giving the bot the answer message twice.

## Known limitation

Wappi latest25 still has no manual `expected_frame` gold. The new bundle is valid for route/text no-op, model-call count, telemetry coverage, and mismatch review. It is not yet sufficient to claim SemanticFrame field accuracy or to migrate regex understanding.
