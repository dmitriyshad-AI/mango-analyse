# ADR-003 F2i Implementation Notes

## Что сделано

Добавлен report-only инструмент `scripts/report_adr003_frame_calibration_queue.py`.

Он объединяет уже существующие источники:

- gold calibration;
- over-handoff levers;
- manager-only exact-proof root cause;
- exact-proof injection shadow.

Цель отчёта — не выбрать кандидатов на включение, а показать очередь работ перед любым active-этапом.

## Главное по сырью 36ea110

- `strict_active_candidates_now=0`.
- `true_frame_too_cautious=14`.
- `true_frame_too_confident=0`.
- `semanticframe_existence_vs_availability=7`.
- `semanticframe_safe_reference_missing_facts=6`.
- `semanticframe_low_confidence=13`.
- `retrieval_delivery_runtime_missing_exact_proof=2`.
- `conversation_plan_scope_missing=2`.
- `policy_manager_only_exact_proof=2`.
- `policy_context_update_exact_proof=2`.

## Инварианты

- Runtime route/text не менялись.
- Direct path/provider/profile/P0-floor не трогались.
- Live Telegram/Wappi/AMO/CRM/Tallanto не трогались.
- Все work item строки несут `active_allowed=false` и `active_block_reason`.

## Почему это следующий шаг

F2h доказал: свежий exact KB proof сам по себе недостаточен. Нужно отдельно чинить:

- калибровку SemanticFrame: existence/format не равно availability/enroll;
- доставку exact proof в runtime retrieval;
- product scope / required fact keys в conversation plan/contract;
- отдельную policy-развилку для `manager_only` и `context_update`.

Активное понижение маршрута пока запрещено.
