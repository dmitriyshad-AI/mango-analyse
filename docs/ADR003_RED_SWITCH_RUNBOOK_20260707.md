# ADR-003 Red-Switch Runbook

Date: 2026-07-07

Updated: 2026-08-11. Невызываемые `route_templates`, `reask_read`, `roles_read` и
`TELEGRAM_READING_APPLY_CLASSES` удалены; команды ниже отражают только
текущие ключи.

Purpose: after the consolidated M1 exam, disable a red ADR-003 class in one small commit while leaving green classes intact. This is a prepared rollback path only; no class is disabled by this document.

## Rule

1. Do not edit the frozen M1 package `a246ece2` or its OUT.
2. Disable only the class that Fable marks red.
3. Keep P0, brand, payment, PII, live-seat promise and output floors intact.
4. Commit the disable patch separately from any cleanup or new feature work.

## Command

List supported switches:

```bash
python3 scripts/adr003_red_switch_plan.py --list
```

Print the exact patch checklist for one class:

```bash
python3 scripts/adr003_red_switch_plan.py fact_select_frame
python3 scripts/adr003_red_switch_plan.py p0_model_led
```

The script is read-only. It prints:

- env flag to force off immediately;
- profile file/symbol to edit in the one disable commit;
- reading class to remove when the red class is semantic-reading based;
- tests to run before the commit.

## Old Profile Baseline

Use this only for a temporary old-profile overlay or B-leg emulation:

```text
TELEGRAM_SEMANTIC_READING_CLASSES=sense_seats,slots_gsf,off_topic,intent_actions,live_status_read
```

Do not use an empty `TELEGRAM_SEMANTIC_READING_CLASSES`: it suppresses profile defaults and makes the run non-attributable.

## Supported Keys

- `fact_select_frame`
- `tone_close_frame_veto`
- `p0_model_led`
- `prose_model_led`
- `payment_refund_dispute_split`
- `seats_default_open`
- `p0_latch_autorelease_v2`
