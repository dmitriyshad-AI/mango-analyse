# TZ-121 Блок B: смысловая проверка

Артефакт: shadow-режим для исправления ложного `won_paid_or_active` в исходах Tallanto.

Аудитория: внутренний аналитик / CRM-аналитика.

## Вердикт

`PASS_WITH_NOTES` для shadow-этапа перед регрейдом.

## Что прошло

- Исправление сфокусировано на реальной бизнес-ошибке: фразы вида `не оплатил`, `не записался` больше не превращаются в ложный выигранный исход.
- Primary-логика ограничена только разрешённым flip `won_paid_or_active -> known_student_or_lead`.
- `payment_pending` явно не применяется в primary, как требует ТЗ.
- Старое поведение сохраняется при `off`.
- Shadow trace показывает причину решения по каждой строке и не содержит ПДн.

## Блокеры

Для shadow-этапа блокеров нет.

## Неблокирующие риски

- Микро-набор синтетический и маленький. Он проверяет механику и ключевые граничные случаи, но не доказывает качество на полном сырье.
- В `primary` код уже есть, но запускать его до регрейда нельзя.
- Нужен регрейд по trace: особенно строки `won_paid_or_active -> tallanto_match_without_outcome` и `churn_or_refused_after_activity -> won_paid_or_active`, потому что они не входят в allowed primary, но важны для понимания качества shadow.

## Чего не проверяли

- Полные реальные наборы не запускались по ТЗ.
- Модельные вызовы не делались: блок B детерминированный, `llm_calls_total=0`.
- Write-контуры AMO/Tallanto/CRM не проверялись, потому что этот блок их не должен трогать.

## Регрессионные проверки

- `test_outcome_linker_default_off_preserves_legacy_negation_behavior`
- `test_outcome_linker_shadow_reports_allowed_won_to_known_flip_without_changing_label`
- `test_outcome_linker_primary_applies_only_allowed_won_to_known_flip`
- `test_outcome_linker_primary_blocks_payment_pending_flip`
- `test_outcome_linker_negated_refusal_does_not_negate_paid_after_dash`
- `test_tz121_outcome_b_micro_shadow_reports_allowed_and_blocked_flips`

## Следующее действие

Отдать shadow-артефакты на регрейд. После одобрения можно отдельным шагом включать B primary и только потом переходить к E.
