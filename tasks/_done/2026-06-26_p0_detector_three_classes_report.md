# P0 detector: three additional classes

Date: 2026-06-26

Branch: `codex/p0-detector-three-classes`

Base: `4caa5eb`

## Scope

Implemented the P0 recall detector extension from:

`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-25_TZ_P0_detector_3_klassa.md`

No live bot, AMO, Tallanto, CRM, stable runtime, or customer sends were touched.

## What Changed

Updated `src/mango_mvp/channels/p0_recall_spec.py`.

New `refund` detections:

- service exit / cancellation:
  - "снять ребёнка с кружка";
  - "выписать ребёнка с занятий";
  - "отменить запись";
  - "отказаться от занятий/курса/обучения";
- paid shift transfer / refund:
  - "перенести оплаченную смену";
  - "вернуть деньги за смену";
  - "сгорает/пропадает смена/путёвка/оплата за смену".

New `legal` detections:

- contract/document claim with all required parts:
  - document object: contract/document/receipt/check/act;
  - error marker: wrong/incorrect/error/typo/mismatch;
  - legal/identity field: date/name/FIO/passport/requisites/data.

The new cases intentionally route through `refund` or `legal`, not `complaint`, so they are not removed by `TELEGRAM_P0_MODEL_LED`.

## Regression Cases

Added required true-positive cases:

- `Хочу снять ребёнка с кружка.` -> `refund`
- `Можно отказаться и выписать ребёнка с занятий?` -> `refund`
- `Нужно перенести оплаченную смену, или вернёте деньги за смену?` -> `refund`
- `В договоре неверная дата и фамилия ребёнка, исправьте.` -> `legal`

Added one extra true-positive guard:

- `Нужно перенести оплаченную смену.` -> `refund`

Added required benign cases:

- `Можно перенести занятие на другой день?`
- `Перенесите урок со вторника на четверг, пожалуйста.`
- `Подскажите по договору-оферте, где он размещён?`
- `Когда пришлёте договор на подпись?`
- `Хочу перевести ребёнка в группу посильнее.`
- `Можно снять копию договора?`

Added one extra benign guard:

- `Хочу отписаться от рассылки.`

## Verification

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_p0_perifraz.py tests/test_answer_safety_classifier.py
```

Result:

```text
132 passed in 0.81s
```

Manual classifier spot-check:

```text
Хочу снять ребёнка с кружка. -> ('refund',), manager_only=True, blocks_autonomy=True
Можно отказаться и выписать ребёнка с занятий? -> ('refund',), manager_only=True, blocks_autonomy=True
Нужно перенести оплаченную смену, или вернёте деньги за смену? -> ('refund',), manager_only=True, blocks_autonomy=True
В договоре неверная дата и фамилия ребёнка, исправьте. -> ('legal',), manager_only=True, blocks_autonomy=True

Можно перенести занятие на другой день? -> (), manager_only=False, blocks_autonomy=False
Перенесите урок со вторника на четверг, пожалуйста. -> (), manager_only=False, blocks_autonomy=False
Подскажите по договору-оферте, где он размещён? -> (), manager_only=False, blocks_autonomy=False
Когда пришлёте договор на подпись? -> (), manager_only=False, blocks_autonomy=False
Хочу перевести ребёнка в группу посильнее. -> (), manager_only=False, blocks_autonomy=False
Можно снять копию договора? -> (), manager_only=False, blocks_autonomy=False
```

P0 model-led filter check:

```text
Хочу снять ребёнка с кружка. -> codes=('refund',), filtered=('refund',)
Нужно перенести оплаченную смену, или вернёте деньги за смену? -> codes=('refund',), filtered=('refund',)
В договоре неверная дата и фамилия ребёнка, исправьте. -> codes=('legal',), filtered=('legal',)
```

## Semantic Status

Formal pass: yes.

Semantic pass: `PASS_WITH_NOTES` for the scoped detector and benign controls in this worktree.

Not claimed:

- no 363-dialog regrade was run here;
- no live route replay was run here;
- no production verdict is made.

## Boundaries

- P0 `complaint` routing was not used for the new classes.
- P0/model-led filter logic was not changed.
- Brand gate was not changed.
- Numeric verification was not changed.
- Step1/reliable-answerer was not changed.
- AMO/Tallanto/CRM/client sends: `0`.

Verdict "ready for prod" was not made.
