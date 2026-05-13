# Deal-Aware Terminal Reason Policy

Date: 2026-05-13

Purpose: prevent AMO closed-deal writeback from using one generic "lost lead" interpretation. A closed deal reason determines the safe product behavior.

## Classes Found In The 50-Row ROP Preview

The sample `stable_runtime/deal_aware_preview_50_20260512_v4` contains only terminal AMO deals (`Закрыто и не реализовано`). Its loss reasons map to seven policy classes:

| Class | Examples | Action |
|---|---|---|
| `active_client_entity_resolution` | `Действующий клиент` | Find the active contact/deal/phone/Tallanto student. Do not treat as lost sale. |
| `duplicate_entity_resolution` | `Дубль`, `Дубль (объединены карточки)` | Find the canonical card/deal. Do not write to duplicate. |
| `lost_or_not_actual` | `Не актуально`, `Ушел к конкурентам`, `Выбрали репетитора` | Do not create active sales next step without a new fresh signal. |
| `lost_or_not_actual` | `Дорого` | Do not create active sales next step unless a fresh price/discount signal exists. |
| `no_contact_archive` | `Архив`, `нет связи`, `Недозвон` | Do not turn old no-contact status into a fresh sales task. |
| `no_application_wrong_direction` | `Не оставлял заявку` | Block sales writeback until the target client is confirmed. |
| `invalid_or_test_no_action` | `Спам`, `Тест` | Not a real lead; block AI writeback. |
| `future_prospect_reactivation` | `Перспектива`, `Перспектива (не подошло расписание)` | Requires a dated reactivation scenario, not a generic sales next step. |
| `company_side_unavailable` | `Закрыли группу (мы)` | Company-side capacity issue; review alternatives first. |
| `refund_or_postsale_service_review` | `Возврат` | Service/finance context; do not write sales next step. |
| `graduate_or_alumni` | `Выпускник` | Use alumni/repeat-sales policy, not ordinary lead sales. |
| `not_qualified_or_out_of_scope` | `Не квал`, `Жуковский`, `Не подходит формат`, `ШД Жако` | Route to review; no automatic sales action. |
| `ambiguous_other_manual_review` | `Другое` | Manual review; reason is not actionable enough. |
| `terminal_lost_without_loss_reason_requires_manual_review` | empty reason on `Закрыто и не реализовано` | Manual review; do not infer why the deal was closed. |

## Current Coverage On The 50-Row Preview

Policy classification over all 50 examples:

- `active_client_entity_resolution`: 7
- `duplicate_entity_resolution`: 13
- `lost_or_not_actual`: 14
- `no_contact_archive`: 12
- `no_application_wrong_direction`: 2
- `not_qualified_or_out_of_scope`: 3
- `ambiguous_other_manual_review`: 1
- `unknown`: 0 after validating against the AMO loss-reason enum catalog.
- terminal lost deals without extracted reason: 52 in the full AMO read-only audit; these are blocked by `terminal_lost_without_loss_reason_requires_manual_review`.

Counts may sum above 50 because composite AMO reasons such as `Недозвон | Дубль` intentionally trigger more than one policy.

## Product Rule

The main deal-aware writeback must be built on active/current deals. Terminal deals are not a normal sales-writeback target. Terminal deals can only enter one of these queues:

- entity resolution: duplicate/current-client/canonical-card search;
- lost/no-action: truly lost or not actual;
- no-contact/reactivation: archive/no-contact, only after a fresh signal;
- manual review: ambiguous or out-of-scope.

## Implementation

Single source of truth:

- `src/mango_mvp/quality/amo_loss_reason_policy.py`

Consumers:

- `src/mango_mvp/quality/crm_text_quality_detector.py`
- `scripts/build_deal_aware_preview_pack.py`
- `scripts/run_crm_writeback_quality_gate.py`

Regression tests:

- `tests/test_amo_loss_reason_policy.py`
- `tests/test_crm_text_quality_detector.py`

Exit criterion for this class:

- Every known AMO terminal loss reason maps to a policy class.
- Any `unknown` reason is blocked from live writeback and added to this policy with a regression test before broader rollout.
