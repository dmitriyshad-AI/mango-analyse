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

## Локальный provider-smoke

После коммита выполнен отдельный direct-path smoke на 7 existence/format
сценариях без `--semantic-frame-enrich-from`.

Результат:

- dialogs: 7;
- turns: 19;
- hard_gate_failures: 0;
- SemanticFrame frames: 18;
- proof_turns: 15;
- freshness_with_proof: 15.

См. `local_provider_measure/provider_smoke_summary.json`.

Полный `dynamic_dialog_transcripts.jsonl` оставлен локально и не включён в git:
PII-греп нашёл в нём публичные контактные строки из KB, а audit pack
не должен тащить такие строки.
