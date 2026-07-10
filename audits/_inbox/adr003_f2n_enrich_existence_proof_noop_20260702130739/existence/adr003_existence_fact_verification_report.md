# ADR-003 F2c Existence/Format Fact Verification

- Status: `pass`
- Source rev: `6549b5b1`
- Gold rows: `79`
- Gold safe self rows: `32`
- Existence/format rows: `9`
- Current handoff rows: `1`
- Handoff with exact KB evidence: `1`
- Handoff without exact KB evidence: `0`
- Already self with exact KB evidence: `5`
- Already self without exact KB evidence: `2`
- Excluded danger/money/P0 rows: `1`

## Группы

- `handoff_with_exact_kb_evidence`: `1`
- `handoff_without_exact_kb_evidence`: `0`
- `already_self_with_exact_kb_evidence`: `5`
- `already_self_without_exact_kb_evidence`: `2`
- `excluded_danger_money_p0`: `1`
- `not_existence_format`: `23`

## Handoff с KB-доказательством

- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` axes=`grade,program_kind` evidence=`kb_exact`
  - client: Да:) Она очень заинтересовала:) у меня ребёнок закончил 5 класс:)
  - requested: brand=unpk subject= grade=после 5 класса format= program=летняя школа
  - best_fact: lvsh_mendeleevo_2026.directions.fizmat.classes (exact, hits=grade,program_kind)

## Handoff без KB-доказательства


## Acceptance Notes

- There are current handoff rows with exact KB evidence; these are candidates for a future fact-gated shadow, not active use.
- Some already-self existence/format answers lack exact KB evidence in metadata/scorer; improve fact trace before active policy.
- Any active self-answer still requires verified exact facts in runtime metadata, not this offline diagnostic matcher.
