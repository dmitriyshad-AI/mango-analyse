# ADR003 F2n No-op Existence Proof Enrichment

## Изменение

Измерительный путь `--semantic-frame-enrich-from` теперь вызывает
`apply_semantic_frame_existence_proof_shadow()` между posthoc SemanticFrame и
self-answer shadow.

Это правка harness, а не живого runtime-пути. Она нужна, чтобы paired no-op
замер реально проверял F2l existence-proof слой.

## Почему

Аудитор подтвердил дыру: прежний enrichment path добавлял SemanticFrame,
self-answer shadow и decision shadow, но не добавлял
`semantic_frame_existence_proof_shadow`. Поэтому paired eval мог показать
route/text no-op, но не доказывал F2l proof.

## Замер

Команда: `--semantic-frame-enrich-from` на OFF-транскриптах M1-прогона
`adr003_f2_clean_36ea110_20260702`, с флагами:

- `TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW=1`;
- `TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW=1`;
- `TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW=1`;
- `TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW=1`;
- `TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE=0`.

Результат:

- dialogs: 131;
- turns: 241;
- route/text/input diff: 0;
- frame present: 241/241;
- frame required fields: 241/241;
- non-frame ON calls: 0;
- self-answer candidates: 0;
- P0/money/operational/manager-only lowered: 0;
- active behavior allowed: false.

## Вывод

F2n доказывает технический shadow-pass для proof-enrichment path, но НЕ даёт
GO на active. Route-only рычага в strict paired no-op замере нет.
