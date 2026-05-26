# GUARDS: что оставить / отключить в v2 / спорно. 2026-05-26

Цель — чтобы старый понимающе-переписывающий каскад НЕ перетирал чистый ответ контракта (§11.5/§13.7).
Классификация из read-only enumeration + правка Кодекса по autonomy_matrix.

## ОСТАВИТЬ — верификаторы безопасности, текст НЕ переписывают
- `apply_payment_confirmation_guard` — блок подтверждения платежа без двух источников.
- `apply_brand_separation_guard` — блок чужого бренда.
- `apply_input_policy_guards` — P0/high-risk во ВВОДЕ → менеджер.
- `apply_funnel_policy_guard` — маршрут на P0/high-risk (текст не трогает).

## ОСТАВИТЬ как safety-fallback, НО выход ОБЯЗАН проходить ре-верификацию v2
(они переписывают draft_text на safe-текст; в v2 этот safe-текст должен проходить тот же verifier, иначе это снова старый шаблонный слой)
- `apply_unstated_subject_guard` — блок неназванного предмета. (В v2 во многом покрыт слотами-с-источником; держать как backstop.)
- `apply_unsupported_promise_guard` — блок чисел/сумм/сроков без факта.
- `apply_unconfirmed_operational_specificity_guard` — блок дат/расписания/доставки без факта.

## autonomy_matrix — НЕ переносить целиком (правка Кодекса)
- ОСТАВИТЬ: route/permission-часть (блок автономии при риске → менеджер).
- ОТКЛЮЧИТЬ: любые замены `draft_text` (`_live_status_manager_check_text`, `_promoted_verified_fact_text`) — нарушают §13.7 →
  должны стать finding/safety-fallback с ре-верификацией внутри нового verifier.

## ОТКЛЮЧИТЬ в v2 — понимающе-переписывающий каскад (его роль забрал контракт)
- `apply_conversation_intent_plan_guard` — выравнивание темы/маршрута + шаблон. Заменён контрактом-планом.
- `apply_known_context_redundant_question_guard` — переспрос известного. Заменён known_slots контракта.
- `apply_answer_quality_rewriter` — переписывание (игнор/повтор/шаблонность). Заменён draft + form-check + X2-тепло v2.
- `apply_humanity_x2_rewriter` — старый X2. ОТКЛЮЧИТЬ, ЛИБО переиспользовать КАК v2-warmth (тогда обязательна ре-верификация выхода).

## СПОРНО — пересмотреть точечно (разделить безопасность и переписывание)
- `apply_high_risk_content_guards` — ОСТАВИТЬ блок риска (P0), УБРАТЬ переписывание под шаблон.
- `apply_humanity_guards` — ОСТАВИТЬ вычистку служебного текста, УБРАТЬ регенерацию/маршрутизацию (это теперь контракт + form-check/X2).

## Принцип проверки
В v2-ветке перечислить все функции, вызываемые после draft_fn, и убедиться: меняют `draft_text` ТОЛЬКО три класса (safety-коррекция/фоллбэк, X2-тепло, sanitize).
Остальные — только finding или route. См. INVARIANT_13_7_PROOF.md.
