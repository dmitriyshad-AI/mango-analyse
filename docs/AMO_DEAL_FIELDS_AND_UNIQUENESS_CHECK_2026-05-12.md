# AMO deal fields and CRM field uniqueness check, 2026-05-12

## Live AMO field check

Read-only check against `https://educent.amocrm.ru` confirmed:

- AMO OAuth/shared DB status: active;
- lead/deal field catalog fetched successfully;
- contact field catalog fetched successfully.

## Deal fields

All expected deal AI fields are present.

| Field | AMO type | Status |
|---|---:|---|
| `AI-сводка по сделке` | textarea | ok |
| `AI-история по сделке` | textarea | ok |
| `AI-рекомендованный следующий шаг` | textarea | ok |
| `AI-дата следующего касания` | text | ok for compact/date value |
| `AI-фактический статус сделки` | textarea | ok |
| `AI-приоритет сделки` | text | ok for compact label |
| `AI-актуальные возражения` | textarea | ok |
| `AI-основание рекомендации` | textarea | ok |
| `AI-качество привязки к сделке` | textarea | ok |
| `AI-предупреждение по сделке` | textarea | ok |
| `AI-Tallanto статус по сделке` | textarea | ok |
| `AI-дата обновления сделки` | date_time | ok |
| `AI-вердикт по закрытию` | text | existing closure-review field |
| `AI-risk: premature close` | text | existing closure-review field |
| `AI-основание вердикта` | textarea | existing closure-review field |

## Contact fields

Current production contact fields are present:

- `Авто история общения` - textarea;
- `Последняя AI-сводка` - textarea;
- `AI-рекомендованный следующий шаг` - textarea;
- `AI-приоритет` - text;
- `Статус матчинга` - text.

The expanded ideal contact fields from `docs/IDEAL_CRM_AI_FIELDS_CONTACT_DEAL_2026-05-12.md` are not created yet:

- `AI-краткая сводка клиента`;
- `AI-история общения`;
- `AI-активные сделки клиента`;
- `AI-учебный контекст Tallanto`;
- `AI-финансы Tallanto`;
- `AI-важные риски клиента`;
- `AI-дата обновления`;
- `AI-источники контекста`.

They are not required for the current contact-writeback stage, but they are useful before productizing the full client profile layer.

## Non-duplication rule

This is now a product-quality rule, not a cosmetic preference.

Each AI field must carry unique information:

- summary fields answer: "what is the current state?";
- history fields answer: "what happened over time?";
- next-step fields answer: "what exactly should the manager do next?";
- priority/status fields answer: "how urgent and what state?";
- Tallanto fields answer: "what is known from study/finance data?";
- warning fields answer: "what can go wrong?".

The same paragraph, sentence, or almost identical set of facts must not be repeated across several fields.

## Implementation status

Implemented in `src/mango_mvp/quality/crm_text_quality_detector.py`:

- risk type: `cross_field_duplicate_information`;
- class id: `Q7`;
- severity: `P2`;
- behavior: blocks live writeback when two CRM AI fields repeat the same manager-facing information.

Regression tests added in `tests/test_crm_text_quality_detector.py`.

CRM writeback quality gate now treats this risk as fail-live in `scripts/run_crm_writeback_quality_gate.py`.

## Test result

- `tests/test_crm_text_quality_detector.py`: 26 passed;
- `tests/test_amo_writeback_guards.py`: 25 passed.
