# ADR-003 F2f Fact-Gated Self-Answer Readiness

- Status: `pass_no_active_candidate`
- Active readiness: `no_go`
- Source rev: `6549b5b1`
- Existence/format rows: `9`
- Current handoff rows: `1`
- Strict F3 draft candidates: `0`
- Manager-only exact-proof needs policy: `1`
- Already self exact proof: `5`
- Blocked no exact proof: `2`
- Excluded danger/money/P0: `1`

## Groups

- `strict_f3_draft_candidate`: `0`
- `manager_only_exact_proof_needs_policy`: `1`
- `already_self_exact_proof`: `5`
- `blocked_no_exact_proof`: `2`
- `excluded_danger_money_p0`: `1`
- `blocked_frame_not_self`: `0`
- `other`: `0`

## Strict F3 Candidates


## Manager-Only Exact-Proof Rows

- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` confidence=`0.78` proof=`kb_exact`
  - product: brand=unpk grade=после 5 класса program=летняя школа subject=
  - fact: `lvsh_mendeleevo_2026.directions.fizmat.classes`
  - blocked: `route_is_manager_only, frame_risk_not_safe, frame_answerability_not_self, frame_must_handoff, low_confidence`

## Acceptance Notes

- No strict draft_for_manager candidates; active F3 remains NO-GO.
- Exact-proof manager_only rows exist; they need separate policy/upstream work and cannot be demoted by F3 route gate.
- Report-only: no route/text/runtime changes.
