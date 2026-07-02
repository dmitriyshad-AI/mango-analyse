# ADR-003 F2c Existence/Format Fact Verification

- Status: `pass`
- Source rev: `896e820`
- Gold rows: `13`
- Gold safe self rows: `8`
- Existence/format rows: `6`
- Current handoff rows: `0`
- Handoff with exact KB evidence: `0`
- Handoff without exact KB evidence: `0`
- Already self with exact KB evidence: `5`
- Already self without exact KB evidence: `0`
- Excluded danger/money/P0 rows: `1`

## Группы

- `handoff_with_exact_kb_evidence`: `0`
- `handoff_without_exact_kb_evidence`: `0`
- `already_self_with_exact_kb_evidence`: `5`
- `already_self_without_exact_kb_evidence`: `0`
- `excluded_danger_money_p0`: `1`
- `not_existence_format`: `2`

## Handoff с KB-доказательством


## Handoff без KB-доказательства


## Acceptance Notes

- No current handoff rows have exact KB evidence in this scorer; route-only active remains a no-go.
- Any active self-answer still requires verified exact facts in runtime metadata, not this offline diagnostic matcher.
