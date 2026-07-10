# ADR-003 F2k Fact-Proof Gap Review

## Что сделано

Сделан report-only пересчёт F2j subset существующими ADR-003 scorers:

- existence/format fact verification;
- fact-gated self-answer readiness;
- exact-proof injection shadow.

Цель: проверить, есть ли после F2j быстрый route-only или prompt-only рычаг для
активного понижения `draft_for_manager` в self-answer.

## Главное

F2j улучшил `requested_action`, но не открыл active-кандидатов.

На пересчитанном subset:

- `strict_f3_draft_candidates=0`;
- `current_handoff_rows=0`;
- `manager_only_exact_proof_rows=0`;
- active readiness остаётся `no_go`.

## Почему это важно

Остаточный пессимизм frame нельзя безопасно лечить простым prompt-bypass:
часть строк не имеет runtime `exact_fact_keys`, часть уже отвечает self-route,
а часть требует отдельной доставки проверенного факта.

## Где смотреть сырьё

- `local_fact_reports/adr003_existence_fact_verification_report.json`;
- `local_fact_reports/adr003_fact_gated_self_answer_readiness_report.json`;
- `local_fact_reports/adr003_exact_proof_injection_shadow_report.json`.
