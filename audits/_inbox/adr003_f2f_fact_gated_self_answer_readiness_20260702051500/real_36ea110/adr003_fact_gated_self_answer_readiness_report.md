# ADR-003 F2f Fact-Gated Self-Answer Readiness

- Status: `pass_no_active_candidate`
- Active readiness: `no_go`
- Source rev: `113c417`
- Existence/format rows: `10`
- Current handoff rows: `2`
- Strict F3 draft candidates: `0`
- Manager-only exact-proof needs policy: `2`
- Already self exact proof: `6`
- Blocked no exact proof: `1`
- Excluded danger/money/P0: `1`

## Groups

- `strict_f3_draft_candidate`: `0`
- `manager_only_exact_proof_needs_policy`: `2`
- `already_self_exact_proof`: `6`
- `blocked_no_exact_proof`: `1`
- `excluded_danger_money_p0`: `1`
- `blocked_frame_not_self`: `0`
- `other`: `0`

## Strict F3 Candidates


## Manager-Only Exact-Proof Rows

- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.86` proof=`kb_exact`
  - product: brand=unpk grade=5 program=летняя школа subject=
  - fact: `lvsh_mendeleevo_2026.directions.fizmat.classes`
  - blocked: `route_is_manager_only, low_confidence`
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` confidence=`0.92` proof=`kb_exact`
  - product: brand=unpk grade=закончил 5-й класс program=летняя смена subject=
  - fact: `lvsh_mendeleevo_2026.directions.fizmat.classes`
  - blocked: `route_is_manager_only, frame_risk_not_safe, frame_answerability_not_self, frame_action_not_safe_self, frame_must_handoff`

## Acceptance Notes

- No strict draft_for_manager candidates; active F3 remains NO-GO.
- Exact-proof manager_only rows exist; they need separate policy/upstream work and cannot be demoted by F3 route gate.
- Report-only: no route/text/runtime changes.
