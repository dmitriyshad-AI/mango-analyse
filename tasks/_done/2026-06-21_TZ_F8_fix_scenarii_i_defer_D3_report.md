# D3 report — F8 fix scenarios and clean defer

Source TZ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_F8_fix_scenarii_i_defer.md`

Branch: `codex/f8-clean-defer-scenarios`

## What changed

1. The M1 target set was rebuilt as `f8_price_axes_selector_20260621_fix_defer/target_set.jsonl`.
2. Scenarios `f8_018_unpk_10_weekday_semester_no_subject` and `f8_030_unpk_subject_independent_informatics` now expect safe clarification / manager draft, not an exact price.
3. The price-axis selector now reads the explicit schedule axis from the user query: weekday vs weekend.
4. New default-off flag: `TELEGRAM_PRICE_AXES_CLEAN_DEFER`.
5. When both `TELEGRAM_PRICE_AXES_SELECTOR=1` and `TELEGRAM_PRICE_AXES_CLEAN_DEFER=1` are enabled, a dead-end price query returns an empty fact pack instead of pulling unrelated facts.
6. Valid price products are not suppressed: the УНПК 5th-grade weekend semester case still returns `37 000 ₽`.
7. `pilot_gold_v1` now enables `TELEGRAM_PRICE_AXES_SELECTOR` and `TELEGRAM_PRICE_AXES_CLEAN_DEFER` by default.
8. Explicit env value `0` still overrides the pilot profile and disables the selector, so rollback is a one-variable change.

## What was not changed

- Knowledge-base facts were not changed.
- No live bot deployment was performed.
- No live systems, CRM, AMO, Tallanto, ASR, Resolve or Analyze were touched.

## Validation

Targeted tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_price_axes_catalog.py
16 passed in 1.44s
```

Pilot profile tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py::test_pilot_gold_v1_enables_full_battle_profile_flags tests/test_subscription_llm_draft_provider.py::test_direct_path_pilot_gold_v1_enables_direct_and_gold_without_extra_flags
2 passed in 1.68s
```

Runtime self-check:

```text
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 - <<'PY'
...
subscription_module_import_ok True
selector_enabled True
clean_defer_enabled True
first_fact_id fact:v3:price_axes_selector:regular_fact_v3_foton_prices_regular_2026_27_online_5_11_class_before_2026_08_01_year_c62eeaa47e_online_year_5_11
first_text_has_price True
PY
```

Full tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3381 passed, 5 skipped, 1 warning in 53.10s
```

Scenario validation:

```text
personas 32
f8_011_unpk_5_weekend_semester_math -> bot_answer_self, 37 000 ₽ expected
f8_018_unpk_10_weekday_semester_no_subject -> draft_for_manager, no exact price expected
f8_030_unpk_subject_independent_informatics -> draft_for_manager, no exact price expected
```

## M1 artifacts

Target set:

```text
/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/f8_price_axes_selector_20260621_fix_defer/target_set.jsonl
```

Set sha256:

```text
c6b1fcc7a3f7b1dd181647f0af43b237ecfa428e5e14920df707a0663e4af7ef
```

Old superseded single-task file was parked, not deleted:

```text
/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1/_parked/superseded_by_f8_fix_20260621/
```

## Current pilot-profile state

`pilot_gold_v1` enables:

- `TELEGRAM_PRICE_AXES_SELECTOR=1`
- `TELEGRAM_PRICE_AXES_CLEAN_DEFER=1`

Rollback:

- set `TELEGRAM_PRICE_AXES_SELECTOR=0`;
- and/or set `TELEGRAM_PRICE_AXES_CLEAN_DEFER=0`.

## Next step

Review the first live drafts with the profile enabled by raw transcript before allowing any auto-send behavior.

Acceptance for M1:

- fabrication in sent text: `0`;
- brand leakage: `0`;
- P0 refund/payment dispute routes to manager;
- `f8_018` / `f8_030` produce clean clarification / manager draft without unrelated olympiad/curator facts;
- `f8_011` still returns `37 000 ₽`;
- summary contains `llm_calls_total`.
