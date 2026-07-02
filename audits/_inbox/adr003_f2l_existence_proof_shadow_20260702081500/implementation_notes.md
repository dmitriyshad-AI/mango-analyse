# ADR-003 F2l Existence-Proof Shadow

## Что изменено

Добавлен default-OFF shadow-слой:

- `TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW`;
- `apply_semantic_frame_existence_proof_shadow()`;
- `metadata.direct_path.semantic_frame_existence_proof_shadow`;
- `existence_proof_shadow_count` в freshness trace self-answer shadow.

Источник proof — существующий
`mango_mvp.knowledge_base.product_existence_axes_catalog`, который строит
структурные axes из KB snapshot.

## Почему так

После F2j стало видно, что “существует ли курс/формат” нельзя решать только
через prompt. Нужно доставить проверяемое доказательство факта в telemetry, а
уже потом мерить, можно ли безопасно понижать draft_for_manager.

## Поведение

Флаг выключен по умолчанию. При выключенном флаге слой no-op.

При включенном флаге меняется только metadata. Route/text/safety_flags не
меняются.
