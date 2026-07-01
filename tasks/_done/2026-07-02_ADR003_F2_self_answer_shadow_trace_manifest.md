# ADR-003 F2 self-answer shadow trace manifest

Дата: 2026-07-02
Ветка: `codex/adr003-semanticframe-migration`
HEAD: `81c02e6`

Назначение: дать Claude #1 и D1 проверяемые указатели на сырьё Ф2 без force-add больших ignored transcripts в git.

## Почему raw transcripts не добавлены в git

Файлы лежат в `audits/_inbox/`, который намеренно игнорируется git. Основные transcripts весят 9-10 МБ и содержат публичные телефоны/почту из клиентских ответов бота (`8 (800) 500-81-51`, `edu@kmipt.ru`). Это не live-ПДн, но без отдельного решения Дмитрия я не превращаю ignored runtime/audit сырьё в tracked repo content.

Вместо этого ниже зафиксированы пути, размеры и SHA256. Для регрейда по сырью надо читать именно эти локальные файлы.

## Source files

| Роль | Путь | Размер | SHA256 |
|---|---:|---:|---|
| F1/v4 ON transcripts | `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/on/dynamic_dialog_transcripts.jsonl` | `9470766` | `9cee88cb906dc1cfce2b3e2d8f889dd22d229a8dfdaa0fd554a5edf8ceb4f290` |
| F1/v4 eval report | `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_semantic_frame_eval_report.json` | `56130` | `68a32f5982ee63a0be3bf4ea9a3127e56779f635b11c9daa3e51e3ea32a3a4fa` |
| F1/v4 gold report | `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_frame_gold_calibration_report.json` | `71974` | `64997c7e598d7391c25731575d377e488109c20935304121f29477d30fa08631` |
| F2 ON transcripts | `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/on/dynamic_dialog_transcripts.jsonl` | `10110606` | `a1274e6f18a6e097224840e85f2f955fe1e3b54f211caecd72f06ae80cd08d9b` |
| F2 eval report | `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/adr003_semantic_frame_eval_report.json` | `58985` | `862b7bf629bfa180ee84e1910241b16748afff8035eb51bda1ab38f0ca4e2a79` |
| F2 candidates | `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/self_answer_shadow_candidates.csv` | `1623` | `763b8a6c4822944caf93d705cd882e1b5d005d65fce22c5b215dd18a97e37a15` |
| F2 gold report | `audits/_inbox/adr003_frame_f2_self_shadow_measure_20260702/gold_calibration/adr003_frame_gold_calibration_report.json` | `75820` | `b4527e5b8f80eeac004ccaa640fca5f0009bfd325811fd809857a68be90cd6f1` |

## Key metrics from F2 eval report

- Compared turns: `241`
- Route/text diff: `0`
- Input diff: `0`
- `would_demote_to_self`: `6`
- `would_demote_by_class`: `price=5`, `format=1`
- `p0_lowered_count`: `0`
- `manager_only_lowered_count`: `0`
- `money_lowered_count`: `0`
- `operational_lowered_count`: `0`
- `freshness_unknown_self_candidates`: `0`
- Blocked: `235`
- Main block reasons: `protected_p0=80`, `route_not_draft_for_manager=114`, `risk_class_not_safe=25`, `low_confidence=9`, `blocking_safety_flags=4`, `deal_stage_blocked=2`, `missing_facts=1`

## Key metrics from F2 gold report

- Gold rows: `79`
- Compared `must_handoff` rows: `77`
- `must_handoff_accuracy`: `0.9351`
- `too_confident`: `0`
- `too_cautious`: `5`
- `p0_misses`: `0`
- confidence bucket `0.90-1.00`: `65/66`, `too_confident=0`

## F2 candidates for semantic reggrade

- `rz_foton_offline_price_06` turn 1, `price`
- `rz_foton_offline_price_06` turn 2, `price`
- `rz_unpk_offline_price_07` turn 1, `price`
- `ra1_foton_platform_and_price` turn 1, `price`
- `ra1_unpk_unknown_slot_price` turn 1, `price`
- `cf142_over_handoff_unpk_clean_ready` turn 1, `format`

## M1 bundle

Папка: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_self_answer_shadow_81c02e6_20260702`

- `PROMT_M1.md`
- `BUNDLE_MANIFEST.json`
- `adr003_f2_self_answer_shadow_81c02e6_overlay.tar.gz`
- overlay SHA256: `c98ae820801ec260109b6c8dea84168d4d018b4fa25e4fb159ea1a338f28d3a1`

## Status

This is `formal_pass` evidence for F2-shadow. It is not `semantic_pass` and not permission for F3-active.

Next gate: Claude #1 must reggrade the 6 candidates and/or the fresh M1 run. F3 active remains NO-GO until Claude #1 reggrade and Дмитрий's separate approval.
