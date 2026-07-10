# ADR-003 F2c Existence/Format Fact Verification

- Status: `pass`
- Source rev: `67e6020`
- Gold rows: `79`
- Gold safe self rows: `32`
- Existence/format rows: `10`
- Current handoff rows: `2`
- Handoff with exact KB evidence: `2`
- Handoff without exact KB evidence: `0`
- Already self with exact KB evidence: `6`
- Already self without exact KB evidence: `1`
- Excluded danger/money/P0 rows: `1`

## Группы

- `handoff_with_exact_kb_evidence`: `2`
- `handoff_without_exact_kb_evidence`: `0`
- `already_self_with_exact_kb_evidence`: `6`
- `already_self_without_exact_kb_evidence`: `1`
- `excluded_danger_money_p0`: `1`
- `not_existence_format`: `22`

## Handoff с KB-доказательством

- `wappi_pair_missing_72h_001#1` route=`manager_only` action=`answer_question` axes=`grade,program_kind` evidence=`kb_exact`
  - client: Да:) Она очень заинтересовала:) у меня ребёнок закончил 5 класс:)
  - requested: brand=unpk subject= grade=5 format= program=летняя школа
  - best_fact: kb_v6_5_client_safe_facts_2026_06_03.city_summer_school_dolgoprudny.client_safe_text (exact, hits=grade,program_kind)
- `wappi_pair_missing_72h_003#1` route=`manager_only` action=`check_availability` axes=`grade,program_kind` evidence=`kb_exact`
  - client: Здравствуйте! Думаю, нет. Это для 7-х, а мой 5-й закончил
  - requested: brand=unpk subject= grade=закончил 5-й класс format= program=летняя смена
  - best_fact: kb_v6_5_client_safe_facts_2026_06_03.city_summer_school_dolgoprudny.client_safe_text (exact, hits=grade,program_kind)

## Handoff без KB-доказательства


## Acceptance Notes

- There are current handoff rows with exact KB evidence; these are candidates for a future fact-gated shadow, not active use.
- Some already-self existence/format answers lack exact KB evidence in metadata/scorer; improve fact trace before active policy.
- Any active self-answer still requires verified exact facts in runtime metadata, not this offline diagnostic matcher.
