# TZ-121 Блок B: primary включён после PASS регрейда

Дата: 2026-06-16

## Решение

Регрейд блока B получил `PASS`. После этого B primary считается разрешённым для офлайн-аналитики.

## Что именно включено

Primary доступен через `outcome_model_mode=primary` / `--outcome-model-mode primary` и применяет только один allowlist flip:

- `won_paid_or_active -> known_student_or_lead`

Что не применяется:

- `won_paid_or_active -> payment_pending`
- любые другие изменения исхода

Default остаётся `off`, то есть старое поведение сохраняется без явного флага.

## Границы

- Только офлайн-аналитика.
- Записей в AMO/Tallanto/CRM нет.
- Полные наборы в этом шаге не запускались.
- Модель не вызывалась, `llm_calls_total=0`.
- `payment_pending` остаётся legacy, как требовал регрейд.

## Проверка

Добавлена регрессионная проверка:

- `test_outcome_linker_cli_accepts_b_primary_mode_without_changing_default`

Существующие проверки блока B:

- `test_outcome_linker_default_off_preserves_legacy_negation_behavior`
- `test_outcome_linker_shadow_reports_allowed_won_to_known_flip_without_changing_label`
- `test_outcome_linker_primary_applies_only_allowed_won_to_known_flip`
- `test_outcome_linker_primary_blocks_payment_pending_flip`

## Следующий шаг

Переход к блоку E: морфология Фотона по корню, shadow на микро-наборе, затем стоп на регрейд.
